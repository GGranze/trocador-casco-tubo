"""Geometria do feixe tubular — equações (15) e (16)-(23) do paper."""
import math

# Tabela 1 do paper: (C, n1) por nº de passes nos tubos, para Pt = 1,25·do
TABELA_NT = {
    "triangular": {1: (0.319, 2.142), 2: (0.249, 2.207), 4: (0.175, 2.285),
                   6: (0.0743, 2.499), 8: (0.0365, 2.675)},
    "quadrado":   {1: (0.215, 2.207), 2: (0.156, 2.291), 4: (0.158, 2.263),
                   6: (0.0402, 2.617), 8: (0.0331, 2.643)},
}


def diametro_interno_tubo(do, razao_di_do=0.8):
    """Eq. (15): di = 0,8·do."""
    return razao_di_do * do


def passo_tubos(do, passo_relativo=1.25):
    """Eq. (18): Pt = 1,25·do."""
    return passo_relativo * do


def folga_c_linha(Pt, do):
    """Eq. (19): C' = Pt - do."""
    return Pt - do


def diametro_hidraulico_casco(do, Pt, arranjo):
    """Eq. (16) quadrado e (17) triangular."""
    if arranjo == "quadrado":
        return 4.0 * (Pt**2 - math.pi * do**2 / 4.0) / (math.pi * do)
    if arranjo == "triangular":
        return 4.0 * (0.43 * Pt**2 - 0.5 * math.pi * do**2 / 4.0) / (0.5 * math.pi * do)
    raise ValueError("arranjo deve ser 'triangular' ou 'quadrado'")


def area_escoamento_casco(Ds, B, do, Pt):
    """Eq. (20): As = Ds·B·(1 - do/Pt)."""
    return Ds * B * (1.0 - do / Pt)


def numero_tubos(Ds, do, n_passes, arranjo, arredondar=True):
    """Eq. (23): Nt = C·(Ds/do)^n1, com C e n1 da Tabela 1."""
    try:
        C, n1 = TABELA_NT[arranjo][n_passes]
    except KeyError:
        raise ValueError(
            f"Sem constantes na Tabela 1 para arranjo='{arranjo}', passes={n_passes}. "
            f"Passes disponíveis: {sorted(TABELA_NT['triangular'])}")
    Nt = C * (Ds / do) ** n1
    return max(1, int(round(Nt))) if arredondar else Nt


def comprimento_tubos(A, do, Nt):
    """L = A/(π·do·Nt).

    O paper não escreve essa equação, mas a usa nas Tabelas 5-7
    (ex. caso 1: 179,3/(π·0,015·1524) = 2,50 m).
    """
    return A / (math.pi * do * Nt)
