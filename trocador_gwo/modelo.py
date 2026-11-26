"""Monta o modelo completo: Especificação (entradas) + Projeto (Ds, do, B) -> Resultado."""
from dataclasses import dataclass, replace, asdict
from typing import Optional
import math

from . import geometria as geo
from . import termica as th
from . import hidraulica as hid
from . import custos as cst
from .fluidos import Fluido, obter_fluido


@dataclass
class Especificacao:
    """Tudo que define o problema (o que NÃO é variável de otimização)."""
    fluido_casco: Fluido
    fluido_tubo: Fluido
    Ti_casco: float                  # [°C]
    To_casco: float
    Ti_tubo: float
    To_tubo: float
    passes_tubo: int = 2
    arranjo: str = "triangular"      # 'triangular' | 'quadrado'
    # Defina q e/ou as vazões. O que faltar é calculado por balanço de energia.
    q: Optional[float] = None        # [W]
    m_casco: Optional[float] = None  # [kg/s]
    m_tubo: Optional[float] = None   # [kg/s]
    # parâmetros do modelo (defaults = paper)
    razao_di_do: float = 0.8         # eq. (15)
    passo_relativo: float = 1.25     # eq. (18)
    p_tubo: float = 2.5              # eq. (29)
    eta_bomba: float = 0.8           # eq. (28)
    b0: float = 0.72                 # eq. (32)
    k_parede: Optional[float] = None  # None = eq. (6) do paper (sem condução)
    # custos
    a1: float = 8000.0
    a2: float = 259.2
    a3: float = 0.91                 # ver nota em custos.py
    custo_energia: float = 0.12      # €/kWh
    horas_ano: float = 7000.0
    anos: int = 10
    taxa: float = 0.10

    def __post_init__(self):
        self.fluido_casco = obter_fluido(self.fluido_casco)
        self.fluido_tubo = obter_fluido(self.fluido_tubo)
        self._fechar_balanco()

    def _fechar_balanco(self):
        fs, ft = self.fluido_casco, self.fluido_tubo
        dTs = abs(self.Ti_casco - self.To_casco)
        dTt = abs(self.To_tubo - self.Ti_tubo)
        if self.q is None:
            if self.m_casco is not None:
                self.q = self.m_casco * fs.cp * dTs
            elif self.m_tubo is not None:
                self.q = self.m_tubo * ft.cp * dTt
            else:
                raise ValueError("Informe q, m_casco ou m_tubo.")
        if self.m_casco is None:
            self.m_casco = self.q / (fs.cp * dTs)
        if self.m_tubo is None:
            self.m_tubo = self.q / (ft.cp * dTt)

    def com(self, **mudancas) -> "Especificacao":
        """Cópia com alterações. Para recalcular vazões, passe m_casco=None/m_tubo=None."""
        return replace(self, **mudancas)


@dataclass
class Resultado:
    # projeto
    Ds: float; do: float; B: float
    # geometria
    di: float; Pt: float; De: float; As: float; Nt: int; L: float
    # térmica
    dtml: float; R: float; P: float; F: float
    Re_t: float; Pr_t: float; Nu_t: float; h_t: float; v_t: float
    Re_s: float; Pr_s: float; Nu_s: float; h_s: float; v_s: float
    U: float; A: float
    # hidráulica
    f_t: float; f_s: float; dP_t: float; dP_s: float; potencia: float
    # custos
    Ci: float; Co: float; Cod: float; Ctot: float
    q: float
    avisos: list

    def __str__(self):
        return (
            f"  Ds={self.Ds:.4f} m  do={self.do*1e3:.2f} mm  B={self.B:.4f} m  L={self.L:.3f} m  Nt={self.Nt}\n"
            f"  vt={self.v_t:.3f} m/s  Re_t={self.Re_t:.0f}  ht={self.h_t:.1f}  |  "
            f"De={self.De*1e3:.2f} mm  Re_s={self.Re_s:.0f}  hs={self.h_s:.1f}\n"
            f"  U={self.U:.1f} W/m²K  A={self.A:.2f} m²  ΔTml={self.dtml:.3f}  F={self.F:.3f}\n"
            f"  ΔPt={self.dP_t/1e3:.2f} kPa  ΔPs={self.dP_s/1e3:.2f} kPa  P={self.potencia:.1f} W\n"
            f"  Ci={self.Ci:,.0f} €  Co={self.Co:,.1f} €/ano  Cod={self.Cod:,.0f} €  Ctot={self.Ctot:,.0f} €"
            + ("".join(f"\n  ⚠ {a}" for a in self.avisos))
        )

    def como_dict(self):
        return asdict(self)


def avaliar(esp: Especificacao, Ds: float, do: float, B: float,
            Nt_inteiro: bool = True) -> Resultado:
    """Calcula o trocador para (Ds, do, B). Lança ProjetoInviavel se não existir solução."""
    fs, ft = esp.fluido_casco, esp.fluido_tubo
    avisos = []

    # --- geometria
    di = geo.diametro_interno_tubo(do, esp.razao_di_do)
    Pt = geo.passo_tubos(do, esp.passo_relativo)
    De = geo.diametro_hidraulico_casco(do, Pt, esp.arranjo)
    As = geo.area_escoamento_casco(Ds, B, do, Pt)
    Nt = geo.numero_tubos(Ds, do, esp.passes_tubo, esp.arranjo, arredondar=Nt_inteiro)

    # --- lado do tubo
    v_t = th.velocidade_tubo(esp.m_tubo, di, ft.rho, esp.passes_tubo, Nt)
    Re_t = th.reynolds_tubo(ft.rho, di, v_t, ft.mu)
    Pr_t = th.prandtl(ft.mu, ft.cp, ft.k)
    Nu_t, h_t = th.h_tubo(Re_t, Pr_t, ft.k, di, th.razao_viscosidade(ft.mu, ft.mu_parede))
    if Re_t < 10000:
        avisos.append(f"Re_tubo={Re_t:.0f} < 10.000: Sieder-Tate é de regime turbulento.")

    # --- lado do casco
    v_s = th.velocidade_casco(esp.m_casco, As, fs.rho)
    Re_s = th.reynolds_casco(esp.m_casco, De, As, fs.mu)
    Pr_s = th.prandtl(fs.mu, fs.cp, fs.k)
    Nu_s, h_s = th.h_casco(Re_s, Pr_s, fs.k, De, th.razao_viscosidade(fs.mu, fs.mu_parede))
    if Re_s < 10000:
        avisos.append(f"Re_casco={Re_s:.0f} < 10.000: correlação de Kern é para Re > 10.000.")
    if Re_s >= 40000:
        avisos.append(f"Re_casco={Re_s:.0f} ≥ 40.000: b0=0,72 do paper vale só para Re < 40.000.")

    # --- térmica global
    U = th.coeficiente_global(h_s, h_t, fs.Rf, ft.Rf, do, di, esp.k_parede)
    dtml = th.dtml_contracorrente(esp.Ti_casco, esp.To_casco, esp.Ti_tubo, esp.To_tubo)
    R, P, F = th.fator_correcao(esp.passes_tubo, esp.Ti_casco, esp.To_casco,
                                esp.Ti_tubo, esp.To_tubo)
    A = th.area_necessaria(esp.q, U, F, dtml)
    L = geo.comprimento_tubos(A, do, Nt)

    # --- hidráulica
    f_t = hid.fator_atrito_tubo(Re_t)
    f_s = hid.fator_atrito_casco(Re_s, esp.b0)
    dP_t = hid.perda_carga_tubo(ft.rho, v_t, L, di, f_t, esp.passes_tubo, esp.p_tubo)
    dP_s = hid.perda_carga_casco(f_s, fs.rho, v_s, L, B, Ds, De)
    Pot = hid.potencia_bombeamento(esp.m_tubo, ft.rho, dP_t, esp.m_casco, fs.rho, dP_s,
                                   esp.eta_bomba)

    # --- custos
    Ci = cst.custo_investimento(A, esp.a1, esp.a2, esp.a3)
    Co = cst.custo_operacao_anual(Pot, esp.custo_energia, esp.horas_ano)
    Cod = cst.custo_operacao_descontado(Co, esp.anos, esp.taxa)
    Ctot = cst.custo_total(Ci, Cod)

    return Resultado(Ds, do, B, di, Pt, De, As, Nt, L, dtml, R, P, F,
                     Re_t, Pr_t, Nu_t, h_t, v_t, Re_s, Pr_s, Nu_s, h_s, v_s,
                     U, A, f_t, f_s, dP_t, dP_s, Pot, Ci, Co, Cod, Ctot, esp.q, avisos)
