"""Propriedades dos fluidos.

Os seis primeiros fluidos vêm direto das Tabelas 2, 3 e 4 do paper
(propriedades na temperatura média entre entrada e saída).

Os fluidos marcados com "(aprox.)" NÃO estão no paper: são valores típicos de
tabela, só para você testar trocas de fluido. Confira/substitua pelos valores
da sua temperatura média real (ou crie o seu com `fluido_custom`).
"""
from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class Fluido:
    nome: str
    rho: float                      # massa específica [kg/m³]
    cp: float                       # calor específico [J/kg·K]
    mu: float                       # viscosidade dinâmica [Pa·s]
    k: float                        # condutividade térmica [W/m·K]
    Rf: float                       # resistência de incrustação [m²·K/W]
    mu_parede: Optional[float] = None  # visc. na parede (para (mu/mu_w)^0,14); None -> razão = 1

    def com(self, **mudancas) -> "Fluido":
        return replace(self, **mudancas)


def fluido_custom(nome, rho, cp, mu, k, Rf=0.0002, mu_parede=None) -> Fluido:
    return Fluido(nome, rho, cp, mu, k, Rf, mu_parede)


# ----------------------------------------------------------------- do paper
FLUIDOS = {
    # Tabela 2 (caso 1)
    "metanol":        Fluido("metanol",        750.0, 2840.0, 0.00034, 0.19, 0.00033),
    "agua_salobra":   Fluido("agua_salobra",   995.0, 4200.0, 0.00080, 0.59, 0.00020),
    # Tabela 3 (caso 2)
    "querosene":      Fluido("querosene",      850.0, 2470.0, 0.00040, 0.13, 0.00061),
    "oleo_cru":       Fluido("oleo_cru",       995.0, 2050.0, 0.00358, 0.13, 0.00061),
    # Tabela 4 (caso 3)
    "agua_destilada": Fluido("agua_destilada", 995.0, 4180.0, 0.00080, 0.62, 0.00017),
    "agua_bruta":     Fluido("agua_bruta",     999.0, 4180.0, 0.00092, 0.62, 0.00017),
}

# ------------------------------------------- NÃO estão no paper (aproximados)
FLUIDOS.update({
    "agua_25C (aprox.)":  Fluido("agua_25C (aprox.)",  997.0, 4180.0, 0.00089, 0.607, 0.00018),
    "agua_60C (aprox.)":  Fluido("agua_60C (aprox.)",  983.0, 4185.0, 0.00047, 0.651, 0.00018),
    "etilenoglicol_50% (aprox. 30C)":
                          Fluido("etilenoglicol_50% (aprox. 30C)", 1060.0, 3300.0, 0.0030, 0.39, 0.00020),
})


def obter_fluido(f) -> Fluido:
    """Aceita um Fluido ou o nome de um fluido da biblioteca."""
    if isinstance(f, Fluido):
        return f
    if f in FLUIDOS:
        return FLUIDOS[f]
    raise KeyError(f"Fluido '{f}' não existe. Disponíveis: {list(FLUIDOS)}")
