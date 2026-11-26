"""Liga o modelo ao GWO: otimiza (Ds, do, B) para minimizar Ctot."""
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .gwo import gwo
from .modelo import Especificacao, Resultado, avaliar
from .termica import ProjetoInviavel

PENALIDADE = 1e12

# Limites das variáveis de projeto [m].
# O PAPER escreve 0,1≤Ds≤1,5; 0,05≤do≤0,5; 0,015≤B≤0,051, mas os resultados das
# Tabelas 5-7 (do=0,015; B=0,475/0,5) só são possíveis com do e B TROCADOS.
# Por isso o default abaixo usa do∈[0,015;0,051] e B∈[0,05;0,5].
LIMITES_PADRAO = {"Ds": (0.1, 1.5), "do": (0.015, 0.051), "B": (0.05, 0.5)}


@dataclass(frozen=True)
class RestricoesEngenharia:
    """Restrições opcionais de validade das correlações e prática TEMA.

    Defina qualquer limite como None para desativá-lo individualmente.
    """
    # Validade das correlações do modelo
    re_tubo_min: Optional[float] = 10_000.0
    re_casco_min: Optional[float] = 10_000.0
    re_casco_max: Optional[float] = 40_000.0
    # Proporções geométricas (TEMA) e limite construtivo
    l_sobre_ds_min: Optional[float] = 3.0
    l_sobre_ds_max: Optional[float] = 10.0
    b_sobre_ds_min: Optional[float] = 0.2
    b_sobre_ds_max: Optional[float] = 1.0
    exigir_b_menor_igual_l: bool = True
    l_max: Optional[float] = 6.0
    # Limites operacionais opcionais (desativados no preset padrão)
    v_t_min: Optional[float] = None         # [m/s]
    v_t_max: Optional[float] = None         # [m/s]
    dP_t_max: Optional[float] = None        # [Pa]
    dP_s_max: Optional[float] = None        # [Pa]
    # Parâmetros da penalidade exterior e comportamento ao final
    peso_base: float = 1e6
    peso_quadratico: float = 1e7
    estrito: bool = True                    # True -> lança ProjetoInviavel se restar violação

    @classmethod
    def padrao(cls) -> "RestricoesEngenharia":
        """Preset padrão: validade hidráulica (Re) + proporções TEMA."""
        return cls()

    @classmethod
    def industrial(cls) -> "RestricoesEngenharia":
        """Preset industrial: padrão + controle de velocidade e perda de carga."""
        return cls(v_t_min=0.5, v_t_max=3.0, dP_t_max=70_000.0, dP_s_max=70_000.0)


@dataclass
class SaidaOtimizacao:
    resultado: Resultado
    historico: np.ndarray
    x: np.ndarray
    violacoes: tuple = ()


def _resolver_restricoes(restricoes) -> Optional[RestricoesEngenharia]:
    if restricoes is None or restricoes is False:
        return None
    if restricoes is True:
        return RestricoesEngenharia()
    if isinstance(restricoes, RestricoesEngenharia):
        return restricoes
    raise TypeError("restricoes deve ser None, bool ou RestricoesEngenharia.")


def verificar_restricoes(res: Resultado, restr: RestricoesEngenharia):
    """Avalia as restrições de engenharia e retorna (penalidade, lista_de_violacoes)."""
    violacoes_rel = []
    mensagens = []

    def checar_min(valor: float, minimo: Optional[float], nome: str, fmt: str = ".3g"):
        if minimo is not None and valor < minimo:
            denom = abs(minimo) if minimo != 0 else 1.0
            violacoes_rel.append((minimo - valor) / denom)
            mensagens.append(f"{nome}={valor:{fmt}} < {minimo:{fmt}}")

    def checar_max(valor: float, maximo: Optional[float], nome: str, fmt: str = ".3g"):
        if maximo is not None and valor > maximo:
            denom = abs(maximo) if maximo != 0 else 1.0
            violacoes_rel.append((valor - maximo) / denom)
            mensagens.append(f"{nome}={valor:{fmt}} > {maximo:{fmt}}")

    l_sobre_ds = res.L / res.Ds
    b_sobre_ds = res.B / res.Ds

    checar_min(res.Re_t, restr.re_tubo_min, "Re_t", ".0f")
    checar_min(res.Re_s, restr.re_casco_min, "Re_s", ".0f")
    checar_max(res.Re_s, restr.re_casco_max, "Re_s", ".0f")
    checar_min(l_sobre_ds, restr.l_sobre_ds_min, "L/Ds", ".2f")
    checar_max(l_sobre_ds, restr.l_sobre_ds_max, "L/Ds", ".2f")
    checar_min(b_sobre_ds, restr.b_sobre_ds_min, "B/Ds", ".2f")
    checar_max(b_sobre_ds, restr.b_sobre_ds_max, "B/Ds", ".2f")
    if restr.exigir_b_menor_igual_l and res.B > res.L:
        violacoes_rel.append((res.B - res.L) / max(res.L, 1e-9))
        mensagens.append(f"B={res.B:.3f} > L={res.L:.3f}")
    checar_max(res.L, restr.l_max, "L", ".2f")
    checar_min(res.v_t, restr.v_t_min, "v_t", ".2f")
    checar_max(res.v_t, restr.v_t_max, "v_t", ".2f")
    if restr.dP_t_max is not None and res.dP_t > restr.dP_t_max:
        violacoes_rel.append((res.dP_t - restr.dP_t_max) / max(restr.dP_t_max, 1.0))
        mensagens.append(f"ΔP_t={res.dP_t/1e3:.1f} kPa > {restr.dP_t_max/1e3:.1f} kPa")
    if restr.dP_s_max is not None and res.dP_s > restr.dP_s_max:
        violacoes_rel.append((res.dP_s - restr.dP_s_max) / max(restr.dP_s_max, 1.0))
        mensagens.append(f"ΔP_s={res.dP_s/1e3:.1f} kPa > {restr.dP_s_max/1e3:.1f} kPa")

    if not violacoes_rel:
        return 0.0, []

    pen = restr.peso_base + restr.peso_quadratico * sum(v * v for v in violacoes_rel)
    return min(pen, 0.5 * PENALIDADE), mensagens


def custo_objetivo(esp: Especificacao, x, restricoes=None) -> float:
    Ds, do, B = x
    restr = _resolver_restricoes(restricoes)
    try:
        res = avaliar(esp, Ds, do, B)
    except (ProjetoInviavel, ValueError, ZeroDivisionError, OverflowError):
        return PENALIDADE

    if not np.isfinite(res.Ctot):
        return PENALIDADE
    if restr is None:
        return res.Ctot

    pen, _ = verificar_restricoes(res, restr)
    return min(res.Ctot + pen, 0.9 * PENALIDADE)


def otimizar(
    esp: Especificacao,
    limites=None,
    n_lobos=30,
    max_iter=200,
    variante="classico",
    seed=None,
    verbose=False,
    restricoes=None,
) -> SaidaOtimizacao:
    restr = _resolver_restricoes(restricoes)
    lim = {**LIMITES_PADRAO, **(limites or {})}
    lb = [lim["Ds"][0], lim["do"][0], lim["B"][0]]
    ub = [lim["Ds"][1], lim["do"][1], lim["B"][1]]

    x, f, hist = gwo(
        lambda v: custo_objetivo(esp, v, restr),
        lb, ub, n_lobos, max_iter, variante, seed, verbose,
    )
    if f >= PENALIDADE:
        raise ProjetoInviavel("GWO não achou nenhum projeto viável nesses limites/condições.")

    res = avaliar(esp, *x)
    violacoes = ()
    if restr is not None:
        _, msgs = verificar_restricoes(res, restr)
        violacoes = tuple(msgs)
        if violacoes:
            if restr.estrito:
                detalhe = "; ".join(violacoes)
                raise ProjetoInviavel(
                    f"Restrições de engenharia não atendidas ({detalhe})."
                )
            res.avisos.extend(f"Restrição não atendida: {m}" for m in violacoes)

    return SaidaOtimizacao(res, hist, x, violacoes)