"""Os 3 casos do paper (Tabelas 2, 3 e 4) + projetos de referência/ótimos (Tabelas 5-7)."""
from .fluidos import FLUIDOS
from .modelo import Especificacao


def caso1() -> Especificacao:
    """Metanol (casco) / água salobra (tubo). Tabela 2."""
    return Especificacao(FLUIDOS["metanol"], FLUIDOS["agua_salobra"],
                         Ti_casco=95, To_casco=40, Ti_tubo=25, To_tubo=40,
                         passes_tubo=2, arranjo="triangular",
                         q=4.34e6, m_casco=27.8, m_tubo=68.0)


def caso2() -> Especificacao:
    """Querosene (casco) / óleo cru (tubo). Tabela 3."""
    return Especificacao(FLUIDOS["querosene"], FLUIDOS["oleo_cru"],
                         Ti_casco=199, To_casco=93.3, Ti_tubo=37.8, To_tubo=76.7,
                         passes_tubo=4, arranjo="quadrado",
                         q=1.44e6, m_casco=5.52, m_tubo=18.8)


def caso3() -> Especificacao:
    """Água destilada (casco) / água bruta (tubo). Tabela 4."""
    return Especificacao(FLUIDOS["agua_destilada"], FLUIDOS["agua_bruta"],
                         Ti_casco=33.9, To_casco=29.4, Ti_tubo=23.9, To_tubo=26.7,
                         passes_tubo=2, arranjo="triangular",
                         q=0.46e6, m_casco=22.07, m_tubo=35.31)


CASOS = {1: caso1, 2: caso2, 3: caso3}

# (Ds, do, B) publicados — para validar o modelo (Tabelas 5, 6, 7)
PROJETOS_PAPER = {
    1: {"original": (0.894, 0.020, 0.356), "gwo": (0.7796, 0.015, 0.475)},
    2: {"original": (0.539, 0.025, 0.127), "gwo": (0.727, 0.015, 0.0862)},
    3: {"original": (0.387, 0.013, 0.305), "gwo": (0.605, 0.015, 0.5)},
}

# Valores publicados para comparação: U, A, Ci, Co, Cod, Ctot, Nt, ht, hs, L
REFERENCIA_PAPER = {
    (1, "original"): dict(Nt=918,  vt=0.75,  ht=3812, hs=1573,   U=615,  A=278.6, L=4.83, Ci=51507, Co=2111,  Cod=12973, Ctot=64480),
    (1, "gwo"):      dict(Nt=1524, vt=0.803, ht=3633, hs=1772.3, U=671.9, A=179.3, L=2.5, Ci=39495, Co=779.9, Cod=4792,  Ctot=44287),
    (2, "original"): dict(Nt=158,  vt=1.44,  ht=619,  hs=920,    U=317,  A=61.5,  L=4.88, Ci=19007, Co=1304,  Cod=8012,  Ctot=27020),
    (2, "gwo"):      dict(Nt=1030, vt=0.6482, ht=534.4, hs=747,  U=246,  A=77.6,  L=1.6,  Ci=21590, Co=369,   Cod=2268,  Ctot=18348),
    (3, "original"): dict(Nt=160,  vt=1.76,  ht=6558, hs=5735,   U=1471, A=46.6,  L=4.88, Ci=16549, Co=4466,  Cod=27440, Ctot=43989),
    (3, "gwo"):      dict(Nt=1194, vt=0.7,   ht=3249, hs=4074,   U=1094, A=63,    L=1.30, Ci=19346, Co=208,   Cod=1282,  Ctot=17332),
}
