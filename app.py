"""Aplicativo Interativo — Projeto e Otimização de Trocador de Calor Casco-Tubo (GWO)."""
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from trocador_gwo.casos import CASOS, PROJETOS_PAPER
from trocador_gwo.fluidos import FLUIDOS, fluido_custom
from trocador_gwo.modelo import Especificacao, avaliar
from trocador_gwo.termica import ProjetoInviavel, fator_correcao

try:
    from trocador_gwo.otimizador import (
        RestricoesEngenharia,
        otimizar,
        verificar_restricoes,
    )
    TEM_RESTRICOES = True
except ImportError:
    from trocador_gwo.otimizador import otimizar
    TEM_RESTRICOES = False

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Trocador Casco-Tubo — GWO",
    page_icon="🔥",
    layout="wide",
)

NOMES_CASOS = {
    1: "Caso 1 — Metanol / Água salobra (4,34 MW)",
    2: "Caso 2 — Querosene / Óleo cru (1,44 MW)",
    3: "Caso 3 — Água destilada / Água bruta (0,46 MW)",
}
LISTA_FLUIDOS = list(FLUIDOS.keys())


# ==============================================================================
# FUNÇÕES DE DIAGNÓSTICO GUIADO (AÇÃO + JUSTIFICATIVA)
# ==============================================================================
def limites_seguros_f(Ti_s: float, To_s: float, Ti_t: float, To_t: float):
    """Calcula limites exatos de temperatura para existência do fator F (passes >= 2).

    A condição exata da Eq. (5) é: a + b + sqrt(a² + b²) < 2·D,
    onde a = |Ti_s - To_s|, b = |To_t - Ti_t|, D = |Ti_s - Ti_t|.
    """
    casco_quente = Ti_s > To_s
    a = abs(Ti_s - To_s)
    b = abs(To_t - Ti_t)
    D = abs(Ti_s - Ti_t)
    lim = {}

    if D > a > 0 and (2 * D - a) > 0:
        b_max = 2.0 * D * (D - a) / (2.0 * D - a)
        lim["b_max"] = b_max
        lim["To_t_lim"] = (Ti_t + b_max) if casco_quente else (Ti_t - b_max)

    if D > b > 0 and (2 * D - b) > 0:
        a_max = 2.0 * D * (D - b) / (2.0 * D - b)
        lim["a_max"] = a_max
        lim["To_s_lim"] = (Ti_s - a_max) if casco_quente else (Ti_s + a_max)

    c = abs(To_s - Ti_t)
    if (2 * c - b) > 1e-9 and b > c:
        x_min = 2.0 * c * (b - c) / (2.0 * c - b)
        lim["Ti_s_lim"] = (To_s + x_min) if casco_quente else (To_s - x_min)

    d = abs(Ti_s - To_t)
    if (2 * d - a) > 1e-9 and a > d:
        y_min = 2.0 * d * (a - d) / (2.0 * d - a)
        lim["Ti_t_lim"] = (To_t - y_min) if casco_quente else (To_t + y_min)

    return lim


def diagnosticar_termica(
    Ti_s: float,
    To_s: float,
    Ti_t: float,
    To_t: float,
    passes_tubo: int,
    modo_balanco: str,
    q_calc: float = None,
    cp_t: float = None,
):
    """Retorna lista de diagnósticos bloqueantes com ações específicas e justificativa."""
    problemas = []
    modo_vazoes = modo_balanco == "Fixar entradas, To_casco e ambas vazões (ṁ_s, ṁ_t)"

    # 1. Temperaturas iguais (ΔT = 0)
    if abs(Ti_s - To_s) < 1e-6:
        problemas.append({
            "titulo": "Temperatura de entrada e saída do casco estão iguais",
            "acoes": [
                f"**Aumente Ti casco** ou **abaixe To casco** (para o casco resfriar), ou o inverso se o casco estiver aquecendo (atualmente ambos = {Ti_s:.1f} °C)."
            ],
            "justificativa": "Sem variação de temperatura (ΔT = 0), não existe transferência de calor no lado do casco (q = ṁ·cp·ΔT = 0) e a vazão mássica tende ao infinito.",
            "botoes": [],
        })
        return problemas

    if abs(To_t - Ti_t) < 1e-6:
        problemas.append({
            "titulo": "Temperatura de entrada e saída do tubo estão iguais",
            "acoes": [
                f"**Aumente To tubo** ou **abaixe Ti tubo** (atualmente ambos = {Ti_t:.1f} °C)."
            ],
            "justificativa": "Sem variação de temperatura no fluido dos tubos (ΔT = 0), não há troca térmica válida e o balanço de energia não fecha.",
            "botoes": [],
        })
        return problemas

    casco_quente = Ti_s > To_s
    tubo_aquece = To_t > Ti_t

    # 2. Sentido térmico invertido (ambos aquecendo ou ambos resfriando)
    if casco_quente != tubo_aquece:
        if casco_quente:
            acoes = [
                f"**Aumente To tubo** para acima de {Ti_t:.1f} °C (ou **abaixe Ti tubo** para abaixo de {To_t:.1f} °C)."
            ]
            desc = "ambos os fluidos estão **resfriando** ao mesmo tempo"
        else:
            acoes = [
                f"**Abaixe To tubo** para abaixo de {Ti_t:.1f} °C (ou **aumente Ti tubo** para acima de {To_t:.1f} °C)."
            ]
            desc = "ambos os fluidos estão **aquecendo** ao mesmo tempo"
        problemas.append({
            "titulo": "Sentido de troca térmica inconsistente (1ª Lei da Termodinâmica)",
            "acoes": acoes,
            "justificativa": f"Com as temperaturas atuais, {desc}. Pela conservação de energia, o calor cedido por um fluido precisa obrigatoriamente ser absorvido pelo outro.",
            "botoes": [],
        })
        return problemas

    # 3. Cruzamento nas extremidades (2ª Lei em contracorrente)
    if casco_quente:
        # Fluido quente = casco; fluido frio = tubo
        if Ti_t >= To_s:
            acoes = [
                f"**Temperatura de entrada do fluido frio (Ti tubo = {Ti_t:.1f} °C) muito alta:** abaixe Ti tubo para **abaixo de {To_s:.1f} °C**.",
                f"**Ou** aumente a temperatura de saída do fluido quente (**To casco**) para **acima de {Ti_t:.1f} °C**.",
            ]
            botoes = [
                (f"⚡ Abaixar Ti tubo para {To_s - 5.0:.1f} °C", {"Ti_tubo": round(To_s - 5.0, 1)}),
                (f"⚡ Aumentar To casco para {Ti_t + 5.0:.1f} °C", {"To_casco": round(Ti_t + 5.0, 1)}),
            ]
            problemas.append({
                "titulo": "Cruzamento de temperaturas na extremidade fria (Ti tubo ≥ To casco)",
                "acoes": acoes,
                "justificativa": "Pela 2ª Lei da Termodinâmica, o calor só flui do fluido mais quente para o mais frio. O fluido quente (casco) não pode sair mais frio do que a temperatura em que o fluido frio (tubo) entra.",
                "botoes": botoes,
            })

        if To_t >= Ti_s:
            acoes = []
            botoes = []
            if modo_vazoes and q_calc and cp_t:
                m_t_min = q_calc / (cp_t * max(Ti_s - Ti_t - 1.0, 1.0))
                acoes.append(
                    f"**Vazão do fluido frio (ṁ_t) muito baixa** (gerou To tubo = {To_t:.1f} °C ≥ Ti casco = {Ti_s:.1f} °C): **aumente ṁ_t para acima de {m_t_min:.1f} kg/s** ou **diminua ṁ_s**."
                )
                botoes.append((f"⚡ Aumentar ṁ_t para {m_t_min * 1.15:.1f} kg/s", {"m_tubo": round(m_t_min * 1.15, 2)}))
            else:
                acoes.append(
                    f"**Temperatura de saída do fluido frio (To tubo = {To_t:.1f} °C) muito alta:** abaixe To tubo para **abaixo de {Ti_s:.1f} °C**."
                )
                botoes.append((f"⚡ Abaixar To tubo para {Ti_s - 5.0:.1f} °C", {"To_tubo": round(Ti_s - 5.0, 1)}))
            acoes.append(
                f"**Ou** aumente a temperatura de entrada do fluido quente (**Ti casco**) para **acima de {To_t:.1f} °C**."
            )
            botoes.append((f"⚡ Aumentar Ti casco para {To_t + 5.0:.1f} °C", {"Ti_casco": round(To_t + 5.0, 1)}))
            problemas.append({
                "titulo": "Cruzamento de temperaturas na extremidade quente (To tubo ≥ Ti casco)",
                "acoes": acoes,
                "justificativa": "Pela 2ª Lei da Termodinâmica, o fluido frio (tubo) não pode sair mais quente do que a temperatura máxima com que o fluido quente (casco) entra no trocador.",
                "botoes": botoes,
            })
    else:
        # Fluido frio = casco; fluido quente = tubo
        if Ti_s >= To_t:
            acoes = [
                f"**Temperatura de entrada do fluido frio (Ti casco = {Ti_s:.1f} °C) muito alta:** abaixe Ti casco para **abaixo de {To_t:.1f} °C**.",
                f"**Ou** aumente a temperatura de saída do fluido quente (**To tubo**) para **acima de {Ti_s:.1f} °C**.",
            ]
            botoes = [
                (f"⚡ Abaixar Ti casco para {To_t - 5.0:.1f} °C", {"Ti_casco": round(To_t - 5.0, 1)}),
            ]
            problemas.append({
                "titulo": "Cruzamento de temperaturas na extremidade fria (Ti casco ≥ To tubo)",
                "acoes": acoes,
                "justificativa": "Pela 2ª Lei da Termodinâmica, o fluido quente (tubo) não pode sair a uma temperatura menor ou igual à entrada do fluido frio (casco).",
                "botoes": botoes,
            })
        if To_s >= Ti_t:
            acoes = [
                f"**Temperatura de saída do fluido frio (To casco = {To_s:.1f} °C) muito alta:** abaixe To casco para **abaixo de {Ti_t:.1f} °C**.",
                f"**Ou** aumente a temperatura de entrada do fluido quente (**Ti tubo**) para **acima de {To_s:.1f} °C**.",
            ]
            botoes = [
                (f"⚡ Abaixar To casco para {Ti_t - 5.0:.1f} °C", {"To_casco": round(Ti_t - 5.0, 1)}),
            ]
            problemas.append({
                "titulo": "Cruzamento de temperaturas na extremidade quente (To casco ≥ Ti tubo)",
                "acoes": acoes,
                "justificativa": "Pela 2ª Lei da Termodinâmica, o fluido frio (casco) não pode sair mais quente do que a temperatura de entrada do fluido quente (tubo).",
                "botoes": botoes,
            })

    if problemas:
        return problemas

    # 4. Fator de correção F indefinido (múltiplos passes nos tubos)
    if passes_tubo > 1:
        try:
            fator_correcao(passes_tubo, Ti_s, To_s, Ti_t, To_t)
        except ProjetoInviavel:
            lim = limites_seguros_f(Ti_s, To_s, Ti_t, To_t)
            acoes = [
                f"**Número de passes muito alto ({passes_tubo} passes) para esse cruzamento de temperaturas:** abaixe o número de passes nos tubos para **1 passe** (contracorrente puro)."
            ]
            botoes = [("⚡ Mudar para 1 passe (contracorrente)", {"passes_tubo": 1})]

            if casco_quente:
                if "To_t_lim" in lim:
                    val_seg = round(lim["To_t_lim"] - 1.5, 1)
                    if modo_vazoes and q_calc and cp_t and lim.get("b_max", 0) > 0:
                        m_min = q_calc / (cp_t * lim["b_max"])
                        acoes.append(
                            f"**Vazão do fluido frio (ṁ_t) muito baixa** (gerou To tubo = {To_t:.1f} °C): **aumente ṁ_t para acima de {m_min:.1f} kg/s** (assim To tubo cai para abaixo de {lim['To_t_lim']:.1f} °C)."
                        )
                        botoes.append((f"⚡ Aumentar ṁ_t para {m_min * 1.1:.1f} kg/s", {"m_tubo": round(m_min * 1.1, 2)}))
                    else:
                        acoes.append(
                            f"**Temperatura de saída do fluido frio (To tubo = {To_t:.1f} °C) muito alta:** abaixe To tubo para **abaixo de {lim['To_t_lim']:.1f} °C**."
                        )
                        if val_seg > Ti_t + 0.5:
                            botoes.append((f"⚡ Abaixar To tubo para {val_seg:.1f} °C", {"To_tubo": val_seg}))
                if "To_s_lim" in lim:
                    val_s = round(lim["To_s_lim"] + 1.5, 1)
                    acoes.append(
                        f"**Temperatura de saída do fluido quente (To casco = {To_s:.1f} °C) muito baixa:** aumente To casco para **acima de {lim['To_s_lim']:.1f} °C**."
                    )
                    if val_s < Ti_s - 0.5:
                        botoes.append((f"⚡ Aumentar To casco para {val_s:.1f} °C", {"To_casco": val_s}))
                if "Ti_t_lim" in lim:
                    acoes.append(
                        f"**Temperatura de entrada do fluido frio (Ti tubo = {Ti_t:.1f} °C) muito alta:** abaixe Ti tubo para **abaixo de {lim['Ti_t_lim']:.1f} °C**."
                    )
                if "Ti_s_lim" in lim:
                    acoes.append(
                        f"**Temperatura de entrada do fluido quente (Ti casco = {Ti_s:.1f} °C) muito baixa:** aumente Ti casco para **acima de {lim['Ti_s_lim']:.1f} °C**."
                    )
            else:
                if "To_s_lim" in lim:
                    acoes.append(
                        f"**Temperatura de saída do fluido frio (To casco = {To_s:.1f} °C) muito alta:** abaixe To casco para **abaixo de {lim['To_s_lim']:.1f} °C**."
                    )
                if "To_t_lim" in lim:
                    acoes.append(
                        f"**Temperatura de saída do fluido quente (To tubo = {To_t:.1f} °C) muito baixa:** aumente To tubo para **acima de {lim['To_t_lim']:.1f} °C**."
                    )
                if "Ti_s_lim" in lim:
                    acoes.append(
                        f"**Temperatura de entrada do fluido frio (Ti casco = {Ti_s:.1f} °C) muito alta:** abaixe Ti casco para **abaixo de {lim['Ti_s_lim']:.1f} °C**."
                    )
                if "Ti_t_lim" in lim:
                    acoes.append(
                        f"**Temperatura de entrada do fluido quente (Ti tubo = {Ti_t:.1f} °C) muito baixa:** aumente Ti tubo para **acima de {lim['Ti_t_lim']:.1f} °C**."
                    )

            problemas.append({
                "titulo": f"Cruzamento térmico incompatível com {passes_tubo} passes nos tubos (fator F indefinido)",
                "acoes": acoes,
                "justificativa": (
                    "Em trocadores com 1 passe no casco e múltiplos passes nos tubos (2, 4, 6 ou 8), parte do trajeto ocorre em "
                    "escoamento cocorrente. Isso impede grandes cruzamentos de temperatura (quando o fluido frio sai muito mais quente "
                    "do que a saída do fluido quente), fazendo o fator de correção F da LMTD deixar de existir matematicamente. "
                    "Trocar para 1 passe restaura o contracorrente puro (F = 1), ou ajustar as temperaturas reduz o cruzamento térmico."
                ),
                "botoes": botoes,
            })

    return problemas


def diagnosticar_alertas_operacionais(res, esp: Especificacao):
    """Traduz alertas de Reynolds, TEMA e fator F em recomendações práticas + justificativas."""
    guias = []
    l_ds = res.L / res.Ds
    b_ds = res.B / res.Ds

    if res.F < 0.75:
        guias.append({
            "tipo": "warning",
            "titulo": f"Fator de correção F baixo (F = {res.F:.3f} < 0,75)",
            "acao": (
                f"**Abaixe o número de passes para 1** (contracorrente puro, F = 1,0), ou afaste as temperaturas de saída "
                f"(abaixe To do fluido frio / aumente To do fluido quente)."
            ),
            "justificativa": (
                "Na prática industrial (TEMA), projetos com F < 0,75 operam na região muito íngreme da curva de correção da LMTD: "
                "pequenas variações de processo derrubam drasticamente a eficiência térmica e exigem áreas muito maiores."
            ),
        })

    if res.Re_t < 10_000:
        dica_passes = (
            f"**aumente o número de passes nos tubos** (atualmente {esp.passes_tubo})"
            if esp.passes_tubo < 8
            else "**diminua o diâmetro do casco (Ds)**"
        )
        guias.append({
            "tipo": "warning",
            "titulo": f"Regime nos tubos abaixo do turbulento (Re_tubo = {res.Re_t:,.0f} < 10.000)",
            "acao": (
                f"Velocidade interna muito baixa ({res.v_t:.2f} m/s): {dica_passes}, **diminua Ds**, "
                f"**aumente do** (menos tubos em paralelo) ou **aumente a vazão do tubo / carga térmica**."
            ),
            "justificativa": (
                "A correlação de Sieder & Tate e o fator de atrito interno usados no modelo são válidos apenas para "
                "escoamento plenamente turbulento (Re ≥ 10.000). Abaixo disso, o modelo tende a superestimar h_t e "
                "velocidades lentas favorecem incrustação (fouling) dentro dos tubos."
            ),
        })

    if res.Re_s < 10_000:
        guias.append({
            "tipo": "warning",
            "titulo": f"Regime no casco abaixo do turbulento (Re_casco = {res.Re_s:,.0f} < 10.000)",
            "acao": (
                f"Espaçamento entre chicanas (B = {res.B:.3f} m) ou casco (Ds = {res.Ds:.3f} m) muito grandes: "
                f"**abaixe B** ou **abaixe Ds** (ou aumente a vazão do casco)."
            ),
            "justificativa": (
                "A correlação de Kern usada para o lado do casco assume regime turbulento (Re > 10.000). "
                "Aproximar as chicanas reduz a área transversal de passagem (As), elevando a velocidade e a turbulência."
            ),
        })

    if res.Re_s >= 40_000:
        guias.append({
            "tipo": "warning",
            "titulo": f"Reynolds no casco acima do limite da correlação (Re_casco = {res.Re_s:,.0f} ≥ 40.000)",
            "acao": (
                f"Velocidade no casco muito alta: **aumente o espaçamento entre chicanas (B)** ou **aumente o diâmetro do casco (Ds)**."
            ),
            "justificativa": (
                "A constante b0 = 0,72 adotada na Eq. (32) para o fator de atrito do casco é válida somente para Re_casco < 40.000. "
                "Além disso, velocidades excessivas elevam fortemente a perda de carga e o risco de vibração nos tubos."
            ),
        })

    if l_ds < 3.0:
        dica_p = (
            f"**abaixe o número de passes nos tubos** (atualmente {esp.passes_tubo}), "
            if esp.passes_tubo > 2
            else ""
        )
        guias.append({
            "tipo": "warning",
            "titulo": f"Trocador muito curto e largo — 'efeito panqueca' (L/Ds = {l_ds:.2f} < 3,0)",
            "acao": (
                f"{dica_p}**abaixe o diâmetro do casco (Ds = {res.Ds:.3f} m)** ou **aumente o diâmetro do tubo (do)**."
            ),
            "justificativa": (
                "A norma TEMA recomenda razão L/Ds entre 3 e 10. Cascos muito largos com tubos curtos (como L = "
                f"{res.L:.2f} m e {res.Nt} tubos) encarecem muito os espelhos, cabeçotes e furação — custos reais "
                "que a fórmula simplificada Ci(A) não enxerga."
            ),
        })
    elif l_ds > 10.0:
        guias.append({
            "tipo": "warning",
            "titulo": f"Trocador muito alongado e fino (L/Ds = {l_ds:.2f} > 10,0)",
            "acao": (
                f"**Aumente o diâmetro do casco (Ds = {res.Ds:.3f} m)** ou **diminua o diâmetro do tubo (do)** "
                f"(ou aumente o número de passes para acomodar mais tubos)."
            ),
            "justificativa": (
                "A norma TEMA recomenda L/Ds ≤ 10 para evitar equipamentos excessivamente esbeltos, com alta perda de carga "
                "e difícil limpeza ou remoção do feixe tubular."
            ),
        })

    if res.L > 6.0:
        guias.append({
            "tipo": "warning",
            "titulo": f"Comprimento dos tubos acima do padrão comercial (L = {res.L:.2f} m > 6,0 m)",
            "acao": "**Aumente Ds** ou **abaixe do** para colocar mais tubos em paralelo e encurtar o trocador.",
            "justificativa": (
                "Tubos acima de 6 m (~20 pés) encarecem fabricação, transporte e exigem grande espaço livre na planta "
                "para retirada do feixe em manutenção."
            ),
        })

    if b_ds < 0.2:
        b_min = 0.2 * res.Ds
        guias.append({
            "tipo": "warning",
            "titulo": f"Chicanas muito próximas entre si (B/Ds = {b_ds:.2f} < 0,20)",
            "acao": f"**Aumente o espaçamento entre chicanas (B)** para pelo menos **{b_min:.3f} m** ou **abaixe Ds**.",
            "justificativa": (
                "Pela norma TEMA, B < 0,2·Ds gera perda de carga desproporcional no casco, dificulta a montagem/limpeza "
                "e aumenta vazamentos nas folgas entre tubos e chicanas."
            ),
        })
    elif b_ds > 1.0:
        guias.append({
            "tipo": "warning",
            "titulo": f"Chicanas muito afastadas entre si (B/Ds = {b_ds:.2f} > 1,00)",
            "acao": f"**Abaixe o espaçamento entre chicanas (B)** para no máximo **{res.Ds:.3f} m** ou **aumente Ds**.",
            "justificativa": (
                "Pela norma TEMA, B > Ds faz o fluido escoar mais longitudinalmente do que transversalmente ao feixe, "
                "reduzindo a eficiência térmica e aumentando o risco de vibração mecânica dos tubos sem suporte."
            ),
        })

    if res.B > res.L:
        guias.append({
            "tipo": "warning",
            "titulo": f"Espaçamento entre chicanas maior que o trocador (B = {res.B:.3f} m > L = {res.L:.2f} m)",
            "acao": f"**Abaixe B** para abaixo de {res.L:.2f} m ou **abaixe Ds** para alongar o trocador.",
            "justificativa": "Fisicamente não é possível instalar chicanas com espaçamento maior do que o próprio comprimento dos tubos.",
        })

    return guias


# ==============================================================================
# GERENCIAMENTO DE ESTADO (SINCRONIZAÇÃO DE CASOS E FLUIDOS)
# ==============================================================================
def _preencher_fluido_estado(prefixo: str, f):
    st.session_state[f"{prefixo}_rho"] = float(f.rho)
    st.session_state[f"{prefixo}_cp"] = float(f.cp)
    st.session_state[f"{prefixo}_mu"] = float(f.mu)
    st.session_state[f"{prefixo}_k"] = float(f.k)
    st.session_state[f"{prefixo}_Rf"] = float(f.Rf)


def carregar_caso(num_caso: int):
    esp = CASOS[num_caso]()
    Ds, do, B = PROJETOS_PAPER[num_caso]["gwo"]

    st.session_state["Ds"] = float(Ds)
    st.session_state["do_mm"] = float(do * 1e3)
    st.session_state["B"] = float(B)
    st.session_state["passes_tubo"] = int(esp.passes_tubo)
    st.session_state["arranjo"] = esp.arranjo
    st.session_state["razao_di_do"] = float(esp.razao_di_do)

    st.session_state["Ti_casco"] = float(esp.Ti_casco)
    st.session_state["To_casco"] = float(esp.To_casco)
    st.session_state["Ti_tubo"] = float(esp.Ti_tubo)
    st.session_state["To_tubo"] = float(esp.To_tubo)
    st.session_state["q_mw"] = float(esp.q / 1e6)
    st.session_state["m_casco"] = float(esp.m_casco)
    st.session_state["m_tubo"] = float(esp.m_tubo)

    st.session_state["fluido_casco_nome"] = esp.fluido_casco.nome
    st.session_state["fluido_tubo_nome"] = esp.fluido_tubo.nome
    _preencher_fluido_estado("fs", esp.fluido_casco)
    _preencher_fluido_estado("ft", esp.fluido_tubo)

    st.session_state["custo_energia"] = float(esp.custo_energia)
    st.session_state["horas_ano"] = float(esp.horas_ano)
    st.session_state["anos"] = int(esp.anos)
    st.session_state["taxa"] = float(esp.taxa * 100.0)
    st.session_state["ultima_comparacao"] = None


def on_trocar_caso():
    carregar_caso(st.session_state["caso_selecionado"])


def on_trocar_fluido_casco():
    f = FLUIDOS[st.session_state["fluido_casco_nome"]]
    _preencher_fluido_estado("fs", f)


def on_trocar_fluido_tubo():
    f = FLUIDOS[st.session_state["fluido_tubo_nome"]]
    _preencher_fluido_estado("ft", f)


if "caso_selecionado" not in st.session_state:
    st.session_state["caso_selecionado"] = 1
    carregar_caso(1)

if st.session_state.get("aplicar_geometria") is not None:
    ds_n, do_n, b_n = st.session_state["aplicar_geometria"]
    st.session_state["Ds"] = float(np.clip(ds_n, 0.10, 1.50))
    st.session_state["do_mm"] = float(np.clip(do_n * 1e3, 15.0, 51.0))
    st.session_state["B"] = float(np.clip(b_n, 0.05, 0.50))
    st.session_state["aplicar_geometria"] = None

if st.session_state.get("aplicar_ajuste") is not None:
    for chave_aj, val_aj in st.session_state["aplicar_ajuste"].items():
        st.session_state[chave_aj] = val_aj
    st.session_state["aplicar_ajuste"] = None


# ==============================================================================
# BARRA LATERAL — ENTRADAS DO USUÁRIO
# ==============================================================================
with st.sidebar:
    st.title("⚙️ Parâmetros do Projeto")

    st.selectbox(
        "Carregar caso de referência (artigo)",
        options=[1, 2, 3],
        format_func=lambda k: NOMES_CASOS[k],
        key="caso_selecionado",
        on_change=on_trocar_caso,
    )

    modo_projeto = st.radio(
        "Modo de cálculo",
        options=["Avaliação Manual (Ds, do, B)", "Otimização Automática (GWO)"],
    )

    st.divider()
    st.subheader("1. Geometria do Trocador")

    if modo_projeto == "Avaliação Manual (Ds, do, B)":
        Ds = st.slider("Diâmetro interno do casco — Ds [m]", 0.10, 1.50, step=0.005, key="Ds")
        do_mm = st.slider("Diâmetro externo do tubo — do [mm]", 15.0, 51.0, step=0.5, key="do_mm")
        B = st.slider("Espaçamento entre chicanas — B [m]", 0.05, 0.50, step=0.005, key="B")
        b_tema_min = max(0.05, 0.2 * Ds)
        b_tema_max = min(0.50, Ds)
        st.caption(f"📏 Faixa TEMA sugerida para B (com Ds={Ds:.2f} m): **{b_tema_min:.3f} a {b_tema_max:.3f} m**")
    else:
        Ds = st.session_state["Ds"]
        do_mm = st.session_state["do_mm"]
        B = st.session_state["B"]
        variante_gwo = st.selectbox("Variante GWO", ["classico", "gaussiana", "exponencial", "beta"])
        c1, c2 = st.columns(2)
        n_lobos = c1.number_input("Lobos", 10, 100, 30, step=5)
        max_iter = c2.number_input("Iterações", 30, 500, 150, step=10)
        seed_gwo = st.number_input("Semente (seed)", 0, 9999, 0, step=1)
        if TEM_RESTRICOES:
            usar_restricoes = st.checkbox("Ativar restrições de engenharia (TEMA + Re)", value=True)
            preset_ind = st.checkbox("Usar preset industrial (v_t e ΔP)", value=False, disabled=not usar_restricoes)
            modo_estrito = st.checkbox("Modo estrito (rejeitar se violar)", value=False, disabled=not usar_restricoes)
        else:
            usar_restricoes = False
            preset_ind = False
            modo_estrito = False
            st.warning("Atualize otimizador.py para habilitar RestricoesEngenharia.")

    c1, c2 = st.columns(2)
    passes_tubo = c1.selectbox("Passes (tubo)", [1, 2, 4, 6, 8], key="passes_tubo")
    arranjo = c2.selectbox("Arranjo", ["triangular", "quadrado"], key="arranjo")

    with st.expander("Detalhes construtivos opcionais"):
        razao_di_do = st.slider("Razão di / do", 0.60, 0.95, step=0.01, key="razao_di_do")
        incluir_parede = st.checkbox("Incluir condução na parede do tubo", value=False)
        k_parede = (
            st.number_input("k da parede [W/m·K] (ex.: aço carbono ~50, inox ~16)", 5.0, 400.0, 50.0, step=5.0)
            if incluir_parede
            else None
        )

    st.divider()
    st.subheader("2. Temperaturas e Vazões")
    modo_balanco = st.selectbox(
        "Fechamento do balanço térmico",
        options=[
            "Fixar temperaturas + Carga (q)",
            "Fixar temperaturas + Vazão do casco (ṁ_s)",
            "Fixar entradas, To_casco e ambas vazões (ṁ_s, ṁ_t)",
        ],
    )

    c1, c2 = st.columns(2)
    Ti_casco = c1.number_input("Ti casco [°C]", -50.0, 500.0, step=1.0, format="%.2f", key="Ti_casco")
    To_casco = c2.number_input("To casco [°C]", -50.0, 500.0, step=1.0, format="%.2f", key="To_casco")

    c1, c2 = st.columns(2)
    Ti_tubo = c1.number_input("Ti tubo [°C]", -50.0, 500.0, step=1.0, format="%.2f", key="Ti_tubo")
    if modo_balanco == "Fixar entradas, To_casco e ambas vazões (ṁ_s, ṁ_t)":
        To_tubo = None
        c2.caption("To tubo: calculado pelo balanço")
    else:
        To_tubo = c2.number_input("To tubo [°C]", -50.0, 500.0, step=1.0, format="%.2f", key="To_tubo")

    # Dica em tempo real da faixa segura de To_tubo
    if To_tubo is not None and abs(Ti_casco - To_casco) > 1e-6:
        if Ti_casco > To_casco and To_casco > Ti_tubo:
            if passes_tubo == 1:
                st.caption(f"🌡️ Faixa viável para To tubo (1 passe): **{Ti_tubo + 0.5:.1f} a {Ti_casco - 0.5:.1f} °C**")
            else:
                lim_live = limites_seguros_f(Ti_casco, To_casco, Ti_tubo, To_tubo)
                if "To_t_lim" in lim_live:
                    st.caption(
                        f"🌡️ Faixa viável para To tubo ({passes_tubo} passes): "
                        f"**{Ti_tubo + 0.5:.1f} a {lim_live['To_t_lim'] - 0.5:.1f} °C**"
                    )

    if modo_balanco == "Fixar temperaturas + Carga (q)":
        q_mw = st.number_input("Carga térmica q [MW]", 0.05, 50.0, step=0.1, format="%.3f", key="q_mw")
        m_casco_in = None
        m_tubo_in = None
    elif modo_balanco == "Fixar temperaturas + Vazão do casco (ṁ_s)":
        q_mw = None
        m_casco_in = st.number_input("Vazão casco ṁ_s [kg/s]", 0.1, 500.0, step=0.5, format="%.2f", key="m_casco")
        m_tubo_in = None
    else:
        q_mw = None
        c1, c2 = st.columns(2)
        m_casco_in = c1.number_input("ṁ_s [kg/s]", 0.1, 500.0, step=0.5, format="%.2f", key="m_casco")
        m_tubo_in = c2.number_input("ṁ_t [kg/s]", 0.1, 500.0, step=0.5, format="%.2f", key="m_tubo")

    st.divider()
    st.subheader("3. Fluidos")

    st.selectbox("Fluido do casco", LISTA_FLUIDOS, key="fluido_casco_nome", on_change=on_trocar_fluido_casco)
    with st.expander("Propriedades do fluido do casco"):
        fs_rho = st.number_input("ρ casco [kg/m³]", 10.0, 3000.0, step=10.0, format="%.1f", key="fs_rho")
        fs_cp = st.number_input("cp casco [J/kg·K]", 100.0, 10000.0, step=50.0, format="%.1f", key="fs_cp")
        fs_mu = st.number_input("μ casco [Pa·s]", 0.00001, 1.0, step=0.00005, format="%.5f", key="fs_mu")
        fs_k = st.number_input("k casco [W/m·K]", 0.01, 5.0, step=0.01, format="%.3f", key="fs_k")
        fs_Rf = st.number_input("Rf casco [m²·K/W]", 0.0, 0.01, step=0.00005, format="%.5f", key="fs_Rf")

    st.selectbox("Fluido do tubo", LISTA_FLUIDOS, key="fluido_tubo_nome", on_change=on_trocar_fluido_tubo)
    with st.expander("Propriedades do fluido do tubo"):
        ft_rho = st.number_input("ρ tubo [kg/m³]", 10.0, 3000.0, step=10.0, format="%.1f", key="ft_rho")
        ft_cp = st.number_input("cp tubo [J/kg·K]", 100.0, 10000.0, step=50.0, format="%.1f", key="ft_cp")
        ft_mu = st.number_input("μ tubo [Pa·s]", 0.00001, 1.0, step=0.00005, format="%.5f", key="ft_mu")
        ft_k = st.number_input("k tubo [W/m·K]", 0.01, 5.0, step=0.01, format="%.3f", key="ft_k")
        ft_Rf = st.number_input("Rf tubo [m²·K/W]", 0.0, 0.01, step=0.00005, format="%.5f", key="ft_Rf")

    with st.expander("4. Parâmetros Econômicos"):
        custo_energia = st.number_input("Energia Ce [€/kWh]", 0.01, 2.0, step=0.01, format="%.3f", key="custo_energia")
        horas_ano = st.number_input("Operação [h/ano]", 500.0, 8760.0, step=100.0, format="%.0f", key="horas_ano")
        anos = st.number_input("Vida útil [anos]", 1, 30, step=1, key="anos")
        taxa_pct = st.number_input("Taxa de desconto [%/ano]", 0.0, 30.0, step=0.5, format="%.1f", key="taxa")


# ==============================================================================
# VALIDAÇÃO TERMODINÂMICA PRÉVIA E CÁLCULO PRINCIPAL
# ==============================================================================
st.title("🔥 Simulador e Otimizador de Trocador de Calor Casco-Tubo (GWO)")
st.caption("Baseado no modelo termo-hidráulico e econômico de Vieira, Rieger & Mariani (2017) com suporte a restrições TEMA.")

fluido_s = fluido_custom(st.session_state["fluido_casco_nome"], fs_rho, fs_cp, fs_mu, fs_k, fs_Rf)
fluido_t = fluido_custom(st.session_state["fluido_tubo_nome"], ft_rho, ft_cp, ft_mu, ft_k, ft_Rf)

# Determina To_tubo efetiva para diagnóstico prévio
q_prev = None
if modo_balanco == "Fixar entradas, To_casco e ambas vazões (ṁ_s, ṁ_t)":
    q_prev = m_casco_in * fluido_s.cp * abs(Ti_casco - To_casco)
    sinal_prev = 1.0 if Ti_casco > To_casco else -1.0
    To_tubo_efetiva = Ti_tubo + sinal_prev * q_prev / (m_tubo_in * fluido_t.cp) if abs(Ti_casco - To_casco) > 1e-6 else Ti_tubo
else:
    To_tubo_efetiva = To_tubo

problemas_term = diagnosticar_termica(
    Ti_casco, To_casco, Ti_tubo, To_tubo_efetiva, passes_tubo, modo_balanco, q_prev, fluido_t.cp
)

if problemas_term:
    for idx_p, prob in enumerate(problemas_term):
        st.error(f"### ❌ {prob['titulo']}")
        st.markdown("**O que você pode alterar na barra lateral:**")
        for acao in prob["acoes"]:
            st.markdown(f"- {acao}")
        st.info(f"**Por quê? (Justificativa física):** {prob['justificativa']}")
        if prob["botoes"]:
            cols_btn = st.columns(len(prob["botoes"]))
            for j, (rotulo_b, payload_b) in enumerate(prob["botoes"]):
                if cols_btn[j].button(rotulo_b, key=f"fix_{idx_p}_{j}", type="primary"):
                    st.session_state["aplicar_ajuste"] = payload_b
                    st.rerun()
    st.stop()

try:
    if modo_balanco == "Fixar temperaturas + Carga (q)":
        esp = Especificacao(
            fluido_s, fluido_t, Ti_casco, To_casco, Ti_tubo, To_tubo,
            passes_tubo=passes_tubo, arranjo=arranjo, q=q_mw * 1e6,
            m_casco=None, m_tubo=None, razao_di_do=razao_di_do, k_parede=k_parede,
            custo_energia=custo_energia, horas_ano=horas_ano, anos=int(anos), taxa=taxa_pct / 100.0,
        )
    elif modo_balanco == "Fixar temperaturas + Vazão do casco (ṁ_s)":
        esp = Especificacao(
            fluido_s, fluido_t, Ti_casco, To_casco, Ti_tubo, To_tubo,
            passes_tubo=passes_tubo, arranjo=arranjo, q=None,
            m_casco=m_casco_in, m_tubo=None, razao_di_do=razao_di_do, k_parede=k_parede,
            custo_energia=custo_energia, horas_ano=horas_ano, anos=int(anos), taxa=taxa_pct / 100.0,
        )
    else:
        esp = Especificacao(
            fluido_s, fluido_t, Ti_casco, To_casco, Ti_tubo, To_tubo_efetiva,
            passes_tubo=passes_tubo, arranjo=arranjo, q=q_prev,
            m_casco=m_casco_in, m_tubo=m_tubo_in, razao_di_do=razao_di_do, k_parede=k_parede,
            custo_energia=custo_energia, horas_ano=horas_ano, anos=int(anos), taxa=taxa_pct / 100.0,
        )

    out_gwo = None
    if modo_projeto == "Avaliação Manual (Ds, do, B)":
        Ds_atual, do_atual, B_atual = Ds, do_mm / 1e3, B
        res = avaliar(esp, Ds_atual, do_atual, B_atual)
    else:
        restr_obj = None
        if TEM_RESTRICOES and usar_restricoes:
            restr_obj = (
                RestricoesEngenharia(v_t_min=0.5, v_t_max=3.0, dP_t_max=70_000.0, dP_s_max=70_000.0, estrito=modo_estrito)
                if preset_ind
                else RestricoesEngenharia(estrito=modo_estrito)
            )
        kwargs_opt = dict(
            n_lobos=int(n_lobos),
            max_iter=int(max_iter),
            variante=variante_gwo,
            seed=int(seed_gwo),
        )
        if TEM_RESTRICOES:
            kwargs_opt["restricoes"] = restr_obj
        out_gwo = otimizar(esp, **kwargs_opt)
        res = out_gwo.resultado
        Ds_atual, do_atual, B_atual = res.Ds, res.do, res.B

except (ProjetoInviavel, ValueError, ZeroDivisionError) as err:
    msg_err = str(err)
    st.error(f"### ❌ Não foi possível fechar o projeto no modo atual\n**Detalhe:** {msg_err}")
    if "Restrições de engenharia" in msg_err:
        dicas_estr = []
        if "Re_t=" in msg_err:
            dicas_estr.append(
                f"**Reynolds no tubo muito baixo:** aumente o número de passes nos tubos (atualmente {passes_tubo}) "
                f"ou desmarque **Modo estrito** na barra lateral para ver a solução mais próxima."
            )
        if "L/Ds=" in msg_err and "<" in msg_err:
            dicas_estr.append(
                f"**Trocador muito curto para {passes_tubo} passes:** reduza o número de passes nos tubos (tente 1 ou 2 passes)."
            )
        if "L=" in msg_err:
            dicas_estr.append("**Comprimento acima de 6 m:** reduza a carga térmica ou desmarque **Modo estrito**.")
        if not dicas_estr:
            dicas_estr.append("Desmarque **Modo estrito** na barra lateral para inspecionar a melhor solução encontrada e ver exatamente qual restrição ficou no limite.")
        st.markdown("**O que ajustar:**")
        for d_e in dicas_estr:
            st.markdown(f"- {d_e}")
        st.info(
            "**Por quê?** No **Modo estrito**, o otimizador rejeita o projeto se qualquer regra da norma TEMA ou limite de "
            "Reynolds (Re ≥ 10.000) não puder ser cumprido dentro das dimensões comerciais."
        )
    st.stop()


# ==============================================================================
# FUNÇÕES AUXILIARES PARA SENSIBILIDADE E VARREDURA
# ==============================================================================
METRICAS = {
    "Custo Total — Ctot [€]": ("Ctot", 1.0, "{:,.0f} €"),
    "Custo Inicial — Ci [€]": ("Ci", 1.0, "{:,.0f} €"),
    "Custo Operacional — Cod [€]": ("Cod", 1.0, "{:,.0f} €"),
    "Área de Troca — A [m²]": ("A", 1.0, "{:.2f} m²"),
    "Coeficiente Global — U [W/m²K]": ("U", 1.0, "{:.1f} W/m²K"),
    "Perda de Carga Casco — ΔP_s [kPa]": ("dP_s", 1e-3, "{:.2f} kPa"),
    "Perda de Carga Tubo — ΔP_t [kPa]": ("dP_t", 1e-3, "{:.2f} kPa"),
    "Comprimento dos Tubos — L [m]": ("L", 1.0, "{:.2f} m"),
}


def _sinal_termico(e: Especificacao) -> float:
    return 1.0 if e.Ti_casco > e.To_casco else -1.0


def calcular_tornado(esp_base: Especificacao, ds_b: float, do_b: float, b_b: float, attr: str, escala: float, pct: float):
    r_base = avaliar(esp_base, ds_b, do_b, b_b)
    v_base = getattr(r_base, attr) * escala
    sinal = _sinal_termico(esp_base)
    fs, ft = esp_base.fluido_casco, esp_base.fluido_tubo

    def _m_casco(f):
        ms = esp_base.m_casco * f
        To_s = esp_base.Ti_casco - sinal * esp_base.q / (ms * fs.cp)
        return avaliar(esp_base.com(To_casco=To_s, m_casco=ms), ds_b, do_b, b_b)

    def _m_tubo(f):
        mt = esp_base.m_tubo * f
        To_t = esp_base.Ti_tubo + sinal * esp_base.q / (mt * ft.cp)
        return avaliar(esp_base.com(To_tubo=To_t, m_tubo=mt), ds_b, do_b, b_b)

    testes = [
        ("Diâmetro do casco (Ds)", lambda f: avaliar(esp_base, ds_b * f, do_b, b_b)),
        ("Diâmetro ext. tubo (do)", lambda f: avaliar(esp_base, ds_b, do_b * f, b_b)),
        ("Espaçamento chicanas (B)", lambda f: avaliar(esp_base, ds_b, do_b, b_b * f)),
        ("Carga térmica (q)", lambda f: avaliar(esp_base.com(q=esp_base.q * f, m_casco=None, m_tubo=None), ds_b, do_b, b_b)),
        ("Vazão casco (ṁ_s, q fixo)", _m_casco),
        ("Vazão tubo (ṁ_t, q fixo)", _m_tubo),
        ("Temp. entrada casco (Ti_s)", lambda f: avaliar(esp_base.com(Ti_casco=esp_base.Ti_casco * f, m_casco=None), ds_b, do_b, b_b)),
        ("Temp. entrada tubo (Ti_t)", lambda f: avaliar(esp_base.com(Ti_tubo=esp_base.Ti_tubo * f, m_tubo=None), ds_b, do_b, b_b)),
        ("Viscosidade casco (μ_s)", lambda f: avaliar(esp_base.com(fluido_casco=fs.com(mu=fs.mu * f)), ds_b, do_b, b_b)),
        ("Viscosidade tubo (μ_t)", lambda f: avaliar(esp_base.com(fluido_tubo=ft.com(mu=ft.mu * f)), ds_b, do_b, b_b)),
        ("Condutividade casco (k_s)", lambda f: avaliar(esp_base.com(fluido_casco=fs.com(k=fs.k * f)), ds_b, do_b, b_b)),
        ("Condutividade tubo (k_t)", lambda f: avaliar(esp_base.com(fluido_tubo=ft.com(k=ft.k * f)), ds_b, do_b, b_b)),
        ("Incrustação casco (Rf_s)", lambda f: avaliar(esp_base.com(fluido_casco=fs.com(Rf=fs.Rf * f)), ds_b, do_b, b_b)),
        ("Incrustação tubo (Rf_t)", lambda f: avaliar(esp_base.com(fluido_tubo=ft.com(Rf=ft.Rf * f)), ds_b, do_b, b_b)),
        ("Custo de energia (Ce)", lambda f: avaliar(esp_base.com(custo_energia=esp_base.custo_energia * f), ds_b, do_b, b_b)),
    ]

    linhas = []
    f_menos, f_mais = 1.0 - pct / 100.0, 1.0 + pct / 100.0
    for nome, fn in testes:
        try:
            v_menos = getattr(fn(f_menos), attr) * escala
            v_mais = getattr(fn(f_mais), attr) * escala
            if not (np.isfinite(v_menos) and np.isfinite(v_mais)) or v_base == 0:
                continue
            d_menos = 100.0 * (v_menos - v_base) / v_base
            d_mais = 100.0 * (v_mais - v_base) / v_base
            amplitude = max(d_menos, d_mais, 0.0) - min(d_menos, d_mais, 0.0)
            linhas.append({
                "Parâmetro": nome,
                "Valor (-%)": v_menos,
                "Base": v_base,
                "Valor (+%)": v_mais,
                "Var (-%)": d_menos,
                "Var (+%)": d_mais,
                "Amplitude [%]": amplitude,
            })
        except (ProjetoInviavel, ValueError, ZeroDivisionError, OverflowError):
            continue

    df = pd.DataFrame(linhas)
    if not df.empty:
        df = df.sort_values("Amplitude [%]", ascending=True).reset_index(drop=True)
    return df, v_base


# ==============================================================================
# ABAS DO APLICATIVO
# ==============================================================================
aba1, aba2, aba3, aba4 = st.tabs([
    "📊 Resultados e Diagnóstico",
    "🌪️ Análise Paramétrica (Tornado)",
    "📈 Curvas de Varredura (1D)",
    "🐺 Otimização GWO (Artigo vs. Engenharia)",
])

# ---------------------------------------------------------------------- ABA 1
with aba1:
    st.markdown(
        f"**Balanço térmico fechado:** `q = {esp.q/1e6:.3f} MW` | "
        f"`Casco ({esp.fluido_casco.nome}): {esp.Ti_casco:.1f} → {esp.To_casco:.1f} °C ({esp.m_casco:.2f} kg/s)` | "
        f"`Tubo ({esp.fluido_tubo.nome}): {esp.Ti_tubo:.1f} → {esp.To_tubo:.1f} °C ({esp.m_tubo:.2f} kg/s)`"
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Custo Total (Ctot)", f"{res.Ctot:,.0f} €")
    k2.metric("Investimento (Ci)", f"{res.Ci:,.0f} €")
    k3.metric("Operacional 10a (Cod)", f"{res.Cod:,.0f} €")
    k4.metric("Coef. Global (U)", f"{res.U:.1f} W/m²K")
    k5.metric("Área (A)", f"{res.A:.2f} m²")
    k6.metric("Tubos / Comp.", f"{res.Nt} | {res.L:.2f} m")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Ds / do / B", f"{res.Ds:.3f}m / {res.do*1e3:.1f}mm / {res.B:.3f}m")
    m2.metric("Razão L / Ds", f"{res.L / res.Ds:.2f}")
    m3.metric("Razão B / Ds", f"{res.B / res.Ds:.2f}")
    m4.metric("Re (tubo / casco)", f"{res.Re_t:,.0f} / {res.Re_s:,.0f}")
    m5.metric("ΔP (tubo / casco)", f"{res.dP_t/1e3:.2f} / {res.dP_s/1e3:.2f} kPa")
    m6.metric("Fator F / Bomba", f"F={res.F:.3f} | {res.potencia:.0f} W")

    # Diagnóstico guiado com ação específica e justificativa
    guias = diagnosticar_alertas_operacionais(res, esp)
    if not guias:
        st.success(
            "✅ **Projeto plenamente adequado:** atende ao regime turbulento das correlações "
            "(10.000 ≤ Re < 40.000), fator F ≥ 0,75 e proporções construtivas da norma TEMA."
        )
    else:
        st.markdown(f"#### ⚠️ Pontos de atenção ({len(guias)}) — o que ajustar e por quê")
        for g in guias:
            st.warning(
                f"**{g['titulo']}**\n\n"
                f"🔧 **Como corrigir:** {g['acao']}\n\n"
                f"📘 **Justificativa:** {g['justificativa']}"
            )

    g1, g2 = st.columns(2)
    with g1:
        fig_custos = go.Figure(
            data=[
                go.Bar(
                    x=["Investimento (Ci)", "Operacional 10 anos (Cod)", "Custo Total (Ctot)"],
                    y=[res.Ci, res.Cod, res.Ctot],
                    marker_color=["#2563EB", "#F59E0B", "#10B981"],
                    text=[f"{res.Ci:,.0f} €", f"{res.Cod:,.0f} €", f"{res.Ctot:,.0f} €"],
                    textposition="outside",
                )
            ]
        )
        fig_custos.update_layout(
            title="Composição Econômica do Trocador [€]",
            yaxis_title="Custo [€]",
            height=380,
            margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_custos, use_container_width=True)

    with g2:
        fig_th = make_subplots(specs=[[{"secondary_y": True}]])
        fig_th.add_trace(
            go.Bar(
                name="Coef. Térmico [W/m²K]",
                x=["h_tubo", "h_casco", "U global"],
                y=[res.h_t, res.h_s, res.U],
                marker_color="#0EA5E9",
                text=[f"{res.h_t:.0f}", f"{res.h_s:.0f}", f"{res.U:.0f}"],
                textposition="outside",
            ),
            secondary_y=False,
        )
        fig_th.add_trace(
            go.Bar(
                name="Perda de Carga [kPa]",
                x=["ΔP_tubo", "ΔP_casco"],
                y=[res.dP_t / 1e3, res.dP_s / 1e3],
                marker_color="#EF4444",
                text=[f"{res.dP_t/1e3:.2f}", f"{res.dP_s/1e3:.2f}"],
                textposition="outside",
            ),
            secondary_y=True,
        )
        fig_th.update_layout(
            title="Desempenho Térmico vs. Hidráulico",
            height=380,
            margin=dict(t=50, b=30),
            legend=dict(orientation="h", y=-0.2),
        )
        fig_th.update_yaxes(title_text="Transferência de Calor [W/m²K]", secondary_y=False)
        fig_th.update_yaxes(title_text="Perda de Carga [kPa]", secondary_y=True)
        st.plotly_chart(fig_th, use_container_width=True)

    with st.expander("📋 Ficha Técnica Completa e Download CSV"):
        d = res.como_dict()
        d.pop("avisos", None)
        d["L_sobre_Ds"] = res.L / res.Ds
        d["B_sobre_Ds"] = res.B / res.Ds
        df_ficha = pd.DataFrame([d]).T.rename(columns={0: "Valor"})
        st.dataframe(df_ficha, use_container_width=True)
        st.download_button(
            "⬇️ Baixar ficha técnica (.csv)",
            data=df_ficha.to_csv().encode("utf-8"),
            file_name="resultado_trocador.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------- ABA 2
with aba2:
    st.subheader("Qual parâmetro influencia mais? (Gráfico de Tornado)")
    st.write(
        "Cada parâmetro é variado **isoladamente** em torno do projeto atual "
        "para medir sua influência direta sobre a saída escolhida. "
        "Os parâmetros no topo são os mais influentes."
    )

    c1, c2 = st.columns([2, 1])
    metrica_nome = c1.selectbox("Variável de saída para analisar", list(METRICAS.keys()))
    pct_sens = c2.slider("Perturbação aplicada (± %)", 5, 30, 15, step=5)

    attr_m, escala_m, fmt_m = METRICAS[metrica_nome]
    df_torn, val_base_m = calcular_tornado(esp, Ds_atual, do_atual, B_atual, attr_m, escala_m, pct_sens)

    if df_torn.empty:
        st.warning("Não foi possível calcular a sensibilidade nesse ponto.")
    else:
        campeao = df_torn.iloc[-1]
        st.info(
            f"🏆 **Parâmetro mais influente sobre {metrica_nome}:** "
            f"**{campeao['Parâmetro']}** — variação de ±{pct_sens}% gera uma amplitude total de "
            f"**{campeao['Amplitude [%]']:.1f}%** em relação à base ({fmt_m.format(val_base_m)})."
        )

        fig_torn = go.Figure()
        fig_torn.add_trace(
            go.Bar(
                name=f"Redução (-{pct_sens}%)",
                y=df_torn["Parâmetro"],
                x=df_torn["Var (-%)"],
                orientation="h",
                marker_color="#2563EB",
                text=[f"{v:+.1f}%" for v in df_torn["Var (-%)"]],
                textposition="outside",
            )
        )
        fig_torn.add_trace(
            go.Bar(
                name=f"Aumento (+{pct_sens}%)",
                y=df_torn["Parâmetro"],
                x=df_torn["Var (+%)"],
                orientation="h",
                marker_color="#EA580C",
                text=[f"{v:+.1f}%" for v in df_torn["Var (+%)"]],
                textposition="outside",
            )
        )
        fig_torn.add_vline(x=0, line_width=1.5, line_color="#64748B")
        fig_torn.update_layout(
            title=f"Sensibilidade de {metrica_nome} (Base = {fmt_m.format(val_base_m)})",
            xaxis_title="Variação relativa em relação à base [%]",
            barmode="group",
            bargap=0.25,
            height=560,
            legend=dict(orientation="h", y=1.08),
            margin=dict(l=180, r=40, t=70, b=40),
        )
        st.plotly_chart(fig_torn, use_container_width=True)

        df_tab = df_torn.sort_values("Amplitude [%]", ascending=False).reset_index(drop=True)
        st.dataframe(
            df_tab.style.format({
                "Valor (-%)": "{:,.2f}",
                "Base": "{:,.2f}",
                "Valor (+%)": "{:,.2f}",
                "Var (-%)": "{:+.2f}%",
                "Var (+%)": "{:+.2f}%",
                "Amplitude [%]": "{:.2f}%",
            }),
            use_container_width=True,
        )


# ---------------------------------------------------------------------- ABA 3
with aba3:
    st.subheader("Curvas de Varredura Paramétrica (1D)")
    st.write("Veja o comportamento contínuo de custos, coeficiente global, perdas de carga e Reynolds ao variar um parâmetro.")

    sinal_v = _sinal_termico(esp)
    opcoes_var = {
        "Diâmetro interno do casco — Ds [m]": (0.20, 1.50, Ds_atual, lambda x: avaliar(esp, x, do_atual, B_atual)),
        "Diâmetro externo do tubo — do [mm]": (15.0, 51.0, do_atual * 1e3, lambda x: avaliar(esp, Ds_atual, x / 1e3, B_atual)),
        "Espaçamento entre chicanas — B [m]": (0.05, 0.50, B_atual, lambda x: avaliar(esp, Ds_atual, do_atual, x)),
        "Carga térmica — q [MW]": (
            max(0.1, esp.q / 1e6 * 0.4),
            esp.q / 1e6 * 1.8,
            esp.q / 1e6,
            lambda x: avaliar(esp.com(q=x * 1e6, m_casco=None, m_tubo=None), Ds_atual, do_atual, B_atual),
        ),
        "Vazão mássica casco — ṁ_s [kg/s] (q fixo)": (
            esp.m_casco * 0.5,
            esp.m_casco * 1.8,
            esp.m_casco,
            lambda x: avaliar(
                esp.com(To_casco=esp.Ti_casco - sinal_v * esp.q / (x * esp.fluido_casco.cp), m_casco=x),
                Ds_atual, do_atual, B_atual,
            ),
        ),
        "Vazão mássica tubo — ṁ_t [kg/s] (q fixo)": (
            esp.m_tubo * 0.5,
            esp.m_tubo * 1.8,
            esp.m_tubo,
            lambda x: avaliar(
                esp.com(To_tubo=esp.Ti_tubo + sinal_v * esp.q / (x * esp.fluido_tubo.cp), m_tubo=x),
                Ds_atual, do_atual, B_atual,
            ),
        ),
        "Viscosidade do tubo — μ_t [Pa·s]": (
            esp.fluido_tubo.mu * 0.3,
            esp.fluido_tubo.mu * 3.0,
            esp.fluido_tubo.mu,
            lambda x: avaliar(esp.com(fluido_tubo=esp.fluido_tubo.com(mu=x)), Ds_atual, do_atual, B_atual),
        ),
    }

    var_nome = st.selectbox("Parâmetro no eixo X", list(opcoes_var.keys()))
    vmin, vmax, v_atual, fn_var = opcoes_var[var_nome]

    xs = np.linspace(vmin, vmax, 50)
    dados_v = {k: [] for k in ["x", "Ctot", "Ci", "Cod", "U", "dP_t", "dP_s", "Re_t", "Re_s"]}
    for xv in xs:
        dados_v["x"].append(xv)
        try:
            rv = fn_var(float(xv))
            dados_v["Ctot"].append(rv.Ctot)
            dados_v["Ci"].append(rv.Ci)
            dados_v["Cod"].append(rv.Cod)
            dados_v["U"].append(rv.U)
            dados_v["dP_t"].append(rv.dP_t / 1e3)
            dados_v["dP_s"].append(rv.dP_s / 1e3)
            dados_v["Re_t"].append(rv.Re_t)
            dados_v["Re_s"].append(rv.Re_s)
        except (ProjetoInviavel, ValueError, ZeroDivisionError, OverflowError):
            for k in ["Ctot", "Ci", "Cod", "U", "dP_t", "dP_s", "Re_t", "Re_s"]:
                dados_v[k].append(None)

    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=dados_v["x"], y=dados_v["Ctot"], name="Ctot [€]", line=dict(color="#10B981", width=3)))
    fig_c.add_trace(go.Scatter(x=dados_v["x"], y=dados_v["Ci"], name="Ci [€]", line=dict(color="#2563EB", dash="dash")))
    fig_c.add_trace(go.Scatter(x=dados_v["x"], y=dados_v["Cod"], name="Cod [€]", line=dict(color="#F59E0B", dash="dot")))
    fig_c.add_vline(x=v_atual, line_dash="dash", line_color="#EF4444", annotation_text="Projeto atual")
    fig_c.update_layout(title=f"Custos vs. {var_nome}", xaxis_title=var_nome, yaxis_title="Custo [€]", height=380)
    st.plotly_chart(fig_c, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_u = make_subplots(specs=[[{"secondary_y": True}]])
        fig_u.add_trace(go.Scatter(x=dados_v["x"], y=dados_v["U"], name="U [W/m²K]", line=dict(color="#0EA5E9", width=2.5)), secondary_y=False)
        fig_u.add_trace(go.Scatter(x=dados_v["x"], y=dados_v["dP_s"], name="ΔP_casco [kPa]", line=dict(color="#EF4444")), secondary_y=True)
        fig_u.add_trace(go.Scatter(x=dados_v["x"], y=dados_v["dP_t"], name="ΔP_tubo [kPa]", line=dict(color="#F97316", dash="dot")), secondary_y=True)
        fig_u.add_vline(x=v_atual, line_dash="dash", line_color="#64748B")
        fig_u.update_layout(title="U e Perdas de Carga", xaxis_title=var_nome, height=360)
        fig_u.update_yaxes(title_text="U [W/m²K]", secondary_y=False)
        fig_u.update_yaxes(title_text="ΔP [kPa]", secondary_y=True)
        st.plotly_chart(fig_u, use_container_width=True)

    with c2:
        fig_re = go.Figure()
        fig_re.add_trace(go.Scatter(x=dados_v["x"], y=dados_v["Re_t"], name="Re_tubo", line=dict(color="#6366F1", width=2.5)))
        fig_re.add_trace(go.Scatter(x=dados_v["x"], y=dados_v["Re_s"], name="Re_casco", line=dict(color="#14B8A6", width=2.5)))
        fig_re.add_hline(y=10_000, line_dash="dash", line_color="#EF4444", annotation_text="Limite turbulento (Re = 10.000)")
        fig_re.add_vline(x=v_atual, line_dash="dash", line_color="#64748B")
        fig_re.update_layout(title="Número de Reynolds (Validade das Correlações)", xaxis_title=var_nome, yaxis_title="Re", height=360)
        st.plotly_chart(fig_re, use_container_width=True)


# ---------------------------------------------------------------------- ABA 4
with aba4:
    st.subheader("Otimização via Grey Wolf Optimizer (GWO)")
    st.write("Compare lado a lado o **projeto atual**, o **GWO no Modo Artigo** (sem restrições) e o **GWO no Modo Engenharia** (TEMA + Re ≥ 10.000).")

    c1, c2, c3 = st.columns(3)
    var_cmp = c1.selectbox("Variante GWO (comparativo)", ["classico", "gaussiana", "exponencial", "beta"])
    lobos_cmp = c2.number_input("Número de lobos", 10, 100, 30, step=5, key="lobos_cmp")
    iter_cmp = c3.number_input("Iterações", 30, 400, 150, step=10, key="iter_cmp")

    if st.button("🚀 Rodar Comparativo GWO (Artigo vs. Engenharia)", type="primary"):
        with st.spinner("Executando otimização GWO..."):
            res_dict = {"atual": res}
            try:
                res_dict["artigo"] = otimizar(esp, n_lobos=int(lobos_cmp), max_iter=int(iter_cmp), variante=var_cmp, seed=0)
            except ProjetoInviavel as e:
                res_dict["artigo"] = str(e)

            if TEM_RESTRICOES:
                try:
                    res_dict["engenharia"] = otimizar(
                        esp, n_lobos=int(lobos_cmp), max_iter=int(iter_cmp),
                        variante=var_cmp, seed=0, restricoes=RestricoesEngenharia(estrito=False),
                    )
                except ProjetoInviavel as e:
                    res_dict["engenharia"] = str(e)
            st.session_state["ultima_comparacao"] = res_dict

    cmp_data = st.session_state.get("ultima_comparacao")
    if cmp_data:
        linhas_cmp = []
        for rotulo, chave in [
            ("Projeto Atual", "atual"),
            ("GWO — Modo Artigo", "artigo"),
            ("GWO — Modo Engenharia (TEMA + Re)", "engenharia"),
        ]:
            if chave not in cmp_data:
                continue
            item = cmp_data[chave]
            if isinstance(item, str):
                st.error(f"**{rotulo}:** {item}")
                continue
            r_obj = item if chave == "atual" else item.resultado
            viols = () if chave == "atual" else getattr(item, "violacoes", ())
            linhas_cmp.append({
                "Configuração": rotulo,
                "Ds [m]": r_obj.Ds,
                "do [mm]": r_obj.do * 1e3,
                "B [m]": r_obj.B,
                "Nt": r_obj.Nt,
                "L [m]": r_obj.L,
                "L/Ds": r_obj.L / r_obj.Ds,
                "U [W/m²K]": r_obj.U,
                "A [m²]": r_obj.A,
                "Re_t": r_obj.Re_t,
                "Re_s": r_obj.Re_s,
                "ΔP_t [kPa]": r_obj.dP_t / 1e3,
                "ΔP_s [kPa]": r_obj.dP_s / 1e3,
                "Ci [€]": r_obj.Ci,
                "Cod [€]": r_obj.Cod,
                "Ctot [€]": r_obj.Ctot,
                "Status": "OK" if not viols else f"Violação: {'; '.join(viols)}",
            })

        if linhas_cmp:
            df_cmp = pd.DataFrame(linhas_cmp)
            st.dataframe(
                df_cmp.style.format({
                    "Ds [m]": "{:.3f}", "do [mm]": "{:.1f}", "B [m]": "{:.3f}",
                    "L [m]": "{:.2f}", "L/Ds": "{:.2f}", "U [W/m²K]": "{:.1f}",
                    "A [m²]": "{:.1f}", "Re_t": "{:,.0f}", "Re_s": "{:,.0f}",
                    "ΔP_t [kPa]": "{:.2f}", "ΔP_s [kPa]": "{:.2f}",
                    "Ci [€]": "{:,.0f}", "Cod [€]": "{:,.0f}", "Ctot [€]": "{:,.0f}",
                }),
                use_container_width=True,
            )

            bc1, bc2 = st.columns(2)
            if "artigo" in cmp_data and not isinstance(cmp_data["artigo"], str):
                ra = cmp_data["artigo"].resultado
                if bc1.button("Aplicar geometria do GWO (Artigo) nos sliders"):
                    st.session_state["aplicar_geometria"] = (ra.Ds, ra.do, ra.B)
                    st.rerun()
            if "engenharia" in cmp_data and not isinstance(cmp_data["engenharia"], str):
                re_ = cmp_data["engenharia"].resultado
                if bc2.button("Aplicar geometria do GWO (Engenharia) nos sliders"):
                    st.session_state["aplicar_geometria"] = (re_.Ds, re_.do, re_.B)
                    st.rerun()

            fig_conv = go.Figure()
            if "artigo" in cmp_data and not isinstance(cmp_data["artigo"], str):
                h_art = cmp_data["artigo"].historico
                fig_conv.add_trace(go.Scatter(y=h_art, name="GWO — Modo Artigo", line=dict(color="#2563EB", width=2.5)))
            if "engenharia" in cmp_data and not isinstance(cmp_data["engenharia"], str):
                h_eng = np.where(cmp_data["engenharia"].historico < 5e5, cmp_data["engenharia"].historico, np.nan)
                fig_conv.add_trace(go.Scatter(y=h_eng, name="GWO — Modo Engenharia (região viável)", line=dict(color="#10B981", width=2.5)))
            fig_conv.update_layout(
                title="Curva de Convergência do GWO",
                xaxis_title="Iteração",
                yaxis_title="Custo Total — Ctot [€]",
                height=380,
            )
            st.plotly_chart(fig_conv, use_container_width=True)
    elif out_gwo is not None:
        fig_conv = go.Figure()
        h_cur = np.where(out_gwo.historico < 5e5, out_gwo.historico, np.nan)
        fig_conv.add_trace(go.Scatter(y=h_cur, name="GWO atual", line=dict(color="#2563EB", width=2.5)))
        fig_conv.update_layout(title="Curva de Convergência da Otimização Atual", xaxis_title="Iteração", yaxis_title="Ctot [€]", height=360)
        st.plotly_chart(fig_conv, use_container_width=True)
