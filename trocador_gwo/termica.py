"""Modelo térmico — equações (1)-(14) do paper."""
import math


class ProjetoInviavel(Exception):
    """Cruzamento de temperaturas, F indefinido, etc."""


# ------------------------------------------------------------------ LMTD / F
def dtml_contracorrente(Ti_s, To_s, Ti_t, To_t):
    """Eq. (2). Usa valores absolutos para valer também com fluido frio no casco."""
    d1, d2 = Ti_s - To_t, To_s - Ti_t
    if d1 * d2 <= 0:
        raise ProjetoInviavel("Cruzamento de temperaturas (ΔT nas pontas ≤ 0).")
    d1, d2 = abs(d1), abs(d2)
    if abs(d1 - d2) < 1e-9:
        return d1
    return (d1 - d2) / math.log(d1 / d2)


def razoes_R_P(Ti_s, To_s, Ti_t, To_t):
    """Eqs. (3) e (4), em módulo."""
    R = abs(Ti_s - To_s) / abs(To_t - Ti_t)
    P = abs(To_t - Ti_t) / abs(Ti_s - Ti_t)
    return R, P


def fator_F(R, P):
    """Eq. (5): fator de correção da LMTD (1 passe no casco, 2+ passes pares nos tubos)."""
    s = math.sqrt(R * R + 1.0)
    try:
        if abs(R - 1.0) < 1e-9:                     # limite R -> 1
            num = P * math.sqrt(2.0) / (1.0 - P)
            den = math.log((2 - P * (2 - math.sqrt(2))) / (2 - P * (2 + math.sqrt(2))))
        else:
            num = s / (R - 1.0) * math.log((1 - P) / (1 - P * R))
            den = math.log((2 - P * (R + 1 - s)) / (2 - P * (R + 1 + s)))
        F = num / den
    except (ValueError, ZeroDivisionError):
        raise ProjetoInviavel("F indefinido: P e R fora da região viável (cruzamento de temperaturas).")
    if not (0 < F <= 1.0 + 1e-9):
        raise ProjetoInviavel(f"F fora de (0,1]: {F:.3f}")
    return min(F, 1.0)


def fator_correcao(n_passes_tubo, Ti_s, To_s, Ti_t, To_t):
    """F = 1 para 1 passe (contracorrente puro); eq. (5) para múltiplos passes."""
    R, P = razoes_R_P(Ti_s, To_s, Ti_t, To_t)
    if n_passes_tubo == 1:
        return R, P, 1.0
    return R, P, fator_F(R, P)


# ------------------------------------------------------------- adimensionais
def reynolds_tubo(rho, di, v, mu):
    """Eq. (9)."""
    return rho * di * v / mu


def prandtl(mu, cp, k):
    """Eqs. (10) e (14)."""
    return mu * cp / k


def reynolds_casco(m_casco, De, As, mu):
    """Eq. (13): Re = G·De/μ, com G = ṁ/As."""
    return m_casco * De / (As * mu)


def razao_viscosidade(mu, mu_parede):
    return 1.0 if mu_parede is None else mu / mu_parede


# ----------------------------------------------------------------- convecção
def h_tubo(Re, Pr, k, di, razao_mu=1.0):
    """Eqs. (7)-(8): Sieder & Tate. h = Nu·k/di."""
    Nu = 0.027 * Re**0.8 * Pr ** (1 / 3) * razao_mu**0.14
    return Nu, Nu * k / di


def h_casco(Re, Pr, k, De, razao_mu=1.0):
    """Eqs. (11)-(12): Kern, chicanas segmentadas, Re > 10.000. h = Nu·k/De."""
    Nu = 0.36 * Re**0.55 * Pr ** (1 / 3) * razao_mu**0.14
    return Nu, Nu * k / De


# -------------------------------------------------------------- coef. global
def coeficiente_global(h_s, h_t, Rf_s, Rf_t, do, di, k_parede=None):
    """Eq. (6): 1/U = 1/hs + Rf,s + (do/di)·(Rf,t + 1/ht).

    O paper lista a condução na parede mas a eq. (6) a omite; por padrão
    seguimos a eq. (6). Passe `k_parede` [W/m·K] para somar do·ln(do/di)/(2k).
    """
    inv_U = 1.0 / h_s + Rf_s + (do / di) * (Rf_t + 1.0 / h_t)
    if k_parede:
        inv_U += do * math.log(do / di) / (2.0 * k_parede)
    return 1.0 / inv_U


# ---------------------------------------------------------------------- área
def area_necessaria(q, U, F, dtml):
    """Eq. (1) explicitada: A = q / (U·F·ΔT_ML)."""
    return q / (U * F * dtml)


# ------------------------------------------------------------- velocidades
def velocidade_casco(m_casco, As, rho):
    """Eq. (21)."""
    return m_casco / (As * rho)


def velocidade_tubo(m_tubo, di, rho, n_passes, Nt):
    """Eq. (22): v = ṁ/((π/4)·di²·ρ) · (n/Nt)."""
    return m_tubo / ((math.pi / 4.0) * di**2 * rho) * (n_passes / Nt)
