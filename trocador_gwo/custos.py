"""Função objetivo — equações (24)-(27) do paper."""


def custo_investimento(A, a1=8000.0, a2=259.2, a3=0.91):
    """Eq. (25): Ci = a1 + a2·A^a3  [€].

    ATENÇÃO: o texto do paper diz a3 = 0,77, mas os valores das tabelas
    (ex.: A = 278,6 m² -> Ci = 51.507 €) só são reproduzidos com a3 = 0,91
    (o valor de Caputo et al. 2008 / Patel & Rao 2010). Usamos 0,91 por padrão.
    """
    return a1 + a2 * A**a3


def custo_operacao_anual(P_watts, custo_energia=0.12, horas_ano=7000.0):
    """Eq. (27): Co = P·Ce·H  [€/ano]  (P em W -> kW; Ce em €/kWh)."""
    return P_watts / 1000.0 * custo_energia * horas_ano


def custo_operacao_descontado(Co_anual, anos=10, taxa=0.10):
    """Eq. (26): Cod = Σ_{k=1..n} Co/(1+i)^k  [€]."""
    return sum(Co_anual / (1.0 + taxa) ** k for k in range(1, anos + 1))


def custo_total(Ci, Cod):
    """Eq. (24): Ctot = Ci + Cod."""
    return Ci + Cod
