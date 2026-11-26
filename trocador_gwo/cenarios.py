"""Ferramentas "e se...?": altere um parâmetro, reotimize e compare com o caso base."""
from .fluidos import obter_fluido
from .modelo import Especificacao
from .otimizador import otimizar
from .termica import ProjetoInviavel


def trocar_fluido_tubo(esp: Especificacao, fluido, To_tubo=None, Ti_tubo=None, manter="q"):
    """Troca o fluido dos tubos. manter='q' mantém a carga térmica (vazões são recalculadas)."""
    return _com(esp, manter, fluido_tubo=obter_fluido(fluido), **_nao_nulos(To_tubo=To_tubo, Ti_tubo=Ti_tubo))


def trocar_fluido_casco(esp: Especificacao, fluido, To_casco=None, Ti_casco=None, manter="q"):
    return _com(esp, manter, fluido_casco=obter_fluido(fluido), **_nao_nulos(To_casco=To_casco, Ti_casco=Ti_casco))


def trocar_passes(esp: Especificacao, passes_tubo):
    """Passes nos tubos (Tabela 1: 1, 2, 4, 6 ou 8)."""
    return esp.com(passes_tubo=passes_tubo)


def trocar_arranjo(esp: Especificacao, arranjo):
    return esp.com(arranjo=arranjo)


def trocar_temperaturas(esp: Especificacao, manter="q", **temps):
    """temps: Ti_casco, To_casco, Ti_tubo, To_tubo. manter='q' ou 'vazoes'."""
    return _com(esp, manter, **temps)


def trocar_carga(esp: Especificacao, q):
    """Nova carga térmica [W]; vazões são recalculadas por balanço."""
    return esp.com(q=q, m_casco=None, m_tubo=None)


def _nao_nulos(**kw):
    return {k: v for k, v in kw.items() if v is not None}


def _com(esp, manter, **mud):
    if manter == "q":      # fixa q, recalcula as duas vazões
        return esp.com(m_casco=None, m_tubo=None, **mud)
    if manter == "vazoes":  # fixa vazões, recalcula q a partir do casco (e m_tubo segue q)
        return esp.com(q=None, **mud)
    raise ValueError("manter deve ser 'q' ou 'vazoes'")


# ---------------------------------------------------------------- comparação
COLUNAS = [
    ("A [m²]", "A", "{:.1f}"),
    ("U [W/m²K]", "U", "{:.0f}"),
    ("ht", "h_t", "{:.0f}"),
    ("hs", "h_s", "{:.0f}"),
    ("Nt", "Nt", "{:d}"),
    ("L [m]", "L", "{:.2f}"),
    ("Ds [m]", "Ds", "{:.3f}"),
    ("L/Ds", "L_Ds", "{:.2f}"),
    ("do [mm]", "do", "{:.1f}"),
    ("B [m]", "B", "{:.3f}"),
    ("ΔP_t [kPa]", "dP_t", "{:.1f}"),
    ("ΔP_s [kPa]", "dP_s", "{:.1f}"),
    ("Ci [€]", "Ci", "{:,.0f}"),
    ("Cod [€]", "Cod", "{:,.0f}"),
    ("Ctot [€]", "Ctot", "{:,.0f}"),
]


def comparar(cenarios: dict, otimizar_cada=True, verbose=True, **kw_gwo):
    """cenarios: {'nome': Especificacao}. Reotimiza cada um (GWO) e imprime tabela.

    Passe restricoes=True ou RestricoesEngenharia(...) em kw_gwo para ativar o modo engenharia.
    Retorna {'nome': Resultado | None}. Cenários inviáveis aparecem como None.
    """
    kw = dict(n_lobos=30, max_iter=150, seed=0) | kw_gwo
    out = {}
    for nome, esp in cenarios.items():
        try:
            out[nome] = otimizar(esp, **kw).resultado
        except ProjetoInviavel as e:
            out[nome] = None
            if verbose:
                print(f"[{nome}] inviável: {e}")
    if verbose:
        _imprimir(out)
    return out


def comparar_modos(cenarios: dict, restricoes=True, verbose=True, **kw_gwo):
    """Compara cada cenário no modo artigo (sem restrições) vs. modo engenharia.

    Retorna {'artigo': {nome: Resultado|None}, 'engenharia': {nome: Resultado|None}}.
    """
    kw_base = dict(n_lobos=30, max_iter=150, seed=0) | {
        k: v for k, v in kw_gwo.items() if k != "restricoes"
    }
    res_art, res_eng, motivos_eng = {}, {}, {}

    for nome, esp in cenarios.items():
        try:
            res_art[nome] = otimizar(esp, restricoes=None, **kw_base).resultado
        except ProjetoInviavel:
            res_art[nome] = None

        try:
            res_eng[nome] = otimizar(esp, restricoes=restricoes, **kw_base).resultado
        except ProjetoInviavel as e:
            res_eng[nome] = None
            motivos_eng[nome] = str(e)

    if verbose:
        _imprimir_modos(res_art, res_eng, motivos_eng)
    return {"artigo": res_art, "engenharia": res_eng}


def _fmt(r, attr, f):
    extras = {
        "do": r.do * 1e3,
        "dP_t": r.dP_t / 1e3,
        "dP_s": r.dP_s / 1e3,
        "L_Ds": r.L / r.Ds,
    }
    v = extras[attr] if attr in extras else getattr(r, attr)
    return f.format(int(v) if attr == "Nt" else v)


def _imprimir(res):
    nomes = list(res)
    w0 = max(len(n) for n in nomes + ["cenário"]) + 1
    print(f"{'':{w0}s}" + "".join(f"{c[0]:>13s}" for c in COLUNAS))
    for n in nomes:
        r = res[n]
        if r is None:
            print(f"{n:{w0}s}  (inviável)")
            continue
        print(f"{n:{w0}s}" + "".join(f"{_fmt(r, a, f):>13s}" for _, a, f in COLUNAS))
    avisos = {n: r.avisos for n, r in res.items() if r and r.avisos}
    for n, a in avisos.items():
        for x in a:
            print(f"  ⚠ [{n}] {x}")


def _imprimir_modos(res_art, res_eng, motivos_eng):
    nomes = list(res_art)
    w0 = max(len(n) for n in nomes + ["cenário"]) + 1
    cab = (
        f"{'':{w0}s}"
        f"{'Ctot art.':>12s}{'L/Ds art.':>10s}{'Re_t art.':>11s}  |"
        f"{'Ctot eng.':>12s}{'L/Ds eng.':>10s}{'Re_t eng.':>11s}"
        f"{'ΔCtot [€]':>12s}{'ΔCtot [%]':>10s}"
    )
    print(cab)
    print("-" * len(cab))
    for n in nomes:
        ra, re_ = res_art[n], res_eng[n]
        if ra is None:
            art_txt = f"{'(inviável)':>33s}"
        else:
            art_txt = f"{ra.Ctot:12,.0f}{ra.L/ra.Ds:10.2f}{ra.Re_t:11.0f}"

        if re_ is None:
            eng_txt = f"{'(inviável)':>33s}{'---':>12s}{'---':>10s}"
        else:
            eng_txt = f"{re_.Ctot:12,.0f}{re_.L/re_.Ds:10.2f}{re_.Re_t:11.0f}"
            if ra is not None:
                diff = re_.Ctot - ra.Ctot
                pct = 100.0 * diff / ra.Ctot
                eng_txt += f"{diff:+12,.0f}{pct:+9.1f}%"
            else:
                eng_txt += f"{'---':>12s}{'---':>10s}"

        print(f"{n:{w0}s}{art_txt}  |{eng_txt}")

    for n, msg in motivos_eng.items():
        print(f"  ✖ [{n}] {msg}")


def variacao_percentual(res, base, campos=("A", "U", "h_t", "h_s", "Ctot")):
    """Variação % de cada cenário em relação a base (nome em res)."""
    b = res[base]
    return {
        n: {c: 100 * (getattr(r, c) / getattr(b, c) - 1) for c in campos}
        for n, r in res.items()
        if r is not None and n != base
    }