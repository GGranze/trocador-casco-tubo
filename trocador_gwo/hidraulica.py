"""Perdas de carga e potência de bombeamento — equações (28)-(32) do paper."""
import math


def fator_atrito_tubo(Re_t):
    """Eq. (30): f = [1,82·log10(Re) - 1,64]^-2."""
    return (1.82 * math.log10(Re_t) - 1.64) ** -2


def perda_carga_tubo(rho, v, L, di, f, n_passes, p=2.5):
    """Eq. (29): ΔPt = (ρ v²/2)·(L/di·f + p)·n."""
    return rho * v**2 / 2.0 * (L / di * f + p) * n_passes


def fator_atrito_casco(Re_s, b0=0.72):
    """Eq. (32): fs = 2·b0·Re^-0,15 (b0 = 0,72 para Re < 40.000)."""
    return 2.0 * b0 * Re_s ** -0.15


def perda_carga_casco(f_s, rho, v, L, B, Ds, De):
    """Eq. (31): ΔPs = fs·(ρ v²/2)·(L/B)·(Ds/De)."""
    return f_s * rho * v**2 / 2.0 * (L / B) * (Ds / De)


def potencia_bombeamento(m_t, rho_t, dP_t, m_s, rho_s, dP_s, eta=0.8):
    """Eq. (28): P = (1/η)·(ṁt/ρt·ΔPt + ṁs/ρs·ΔPs)  [W]."""
    return (m_t / rho_t * dP_t + m_s / rho_s * dP_s) / eta
