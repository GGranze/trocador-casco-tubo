"""Exemplos de uso. Rode:  python -m trocador_gwo.exemplo
Edite à vontade — cada bloco é independente.
"""
from .casos import caso1, caso2, caso3
from .cenarios import (
    comparar,
    comparar_modos,
    trocar_arranjo,
    trocar_carga,
    trocar_fluido_tubo,
    trocar_passes,
    trocar_temperaturas,
    variacao_percentual,
)
from .fluidos import fluido_custom
from .modelo import avaliar
from .otimizador import RestricoesEngenharia, otimizar

# ------------------------------------------------------------------ 1) caso base
base = caso1()  # metanol (casco) / água salobra (tubo), 2 passes
out = otimizar(base, n_lobos=30, max_iter=150, seed=0)
print("CASO 1 otimizado (GWO clássico):\n", out.resultado, "\n")

# ------------------------------------------- 2) avaliar um projeto que VOCÊ define
print(
    "Projeto manual (Ds=0,8 m, do=20 mm, B=0,3 m):\n",
    avaliar(base, 0.8, 0.020, 0.3),
    "\n",
)

# ------------------------------------------------- 3) trocar o fluido refrigerante
print("=== Trocar o refrigerante (tubo) ===")
comparar({
    "base: água salobra": base,
    "água 25°C": trocar_fluido_tubo(base, "agua_25C (aprox.)"),
    "etilenoglicol 50%": trocar_fluido_tubo(base, "etilenoglicol_50% (aprox. 30C)"),
    "fluido custom": trocar_fluido_tubo(
        base,
        fluido_custom("meu_fluido", rho=900, cp=3000, mu=0.0012, k=0.4, Rf=0.0003),
    ),
})

# ------------------------------------------------------- 4) trocar nº de passes
print("\n=== Passes nos tubos ===")
comparar({f"{n} passes": trocar_passes(base, n) for n in (1, 2, 4, 6, 8)})

# -------------------------------------------------------- 5) trocar o arranjo
print("\n=== Arranjo ===")
comparar({"triangular": base, "quadrado": trocar_arranjo(base, "quadrado")})

# ----------------------------------------- 6) alterar temperaturas / carga térmica
print("\n=== Temperaturas (mantendo q; vazões recalculadas) ===")
res = comparar({
    "base (To_tubo=40)": base,
    "To_tubo=35": trocar_temperaturas(base, To_tubo=35),
    "To_tubo=50": trocar_temperaturas(base, To_tubo=50),
    "To_casco=50": trocar_temperaturas(base, To_casco=50),
    "Ti_tubo=15": trocar_temperaturas(base, Ti_tubo=15),
})
print("\nVariação % vs base:")
for nome, d in variacao_percentual(res, "base (To_tubo=40)").items():
    print(f"  {nome:14s}", "  ".join(f"{k}: {v:+.1f}%" for k, v in d.items()))

print("\n=== Carga térmica ===")
comparar({
    "q=4,34 MW": base,
    "q=3 MW": trocar_carga(base, 3.0e6),
    "q=6 MW": trocar_carga(base, 6.0e6),
})

# ------------------------------------------------- 7) variantes do GWO (eqs. 37-39)
print("\n=== Variantes do GWO no caso 3 ===")
c3 = caso3()
for v in ("classico", "exponencial", "gaussiana", "beta"):
    r = otimizar(c3, variante=v, seed=1).resultado
    print(f"  {v:12s} Ctot = {r.Ctot:,.0f} €")

# ---------------------------------- 8) Modo Artigo vs. Modo Engenharia (TEMA + Re)
print("\n=== Modo Artigo vs. Modo Engenharia — Passes nos tubos (Caso 1) ===")
comparar_modos({f"{n} passes": trocar_passes(base, n) for n in (1, 2, 4, 6, 8)})

print("\n=== Modo Artigo vs. Modo Engenharia — Troca de refrigerante (Caso 1) ===")
comparar_modos({
    "base: água salobra": base,
    "água 25°C": trocar_fluido_tubo(base, "agua_25C (aprox.)"),
    "etilenoglicol 50%": trocar_fluido_tubo(base, "etilenoglicol_50% (aprox. 30C)"),
    "fluido custom": trocar_fluido_tubo(
        base,
        fluido_custom("meu_fluido", rho=900, cp=3000, mu=0.0012, k=0.4, Rf=0.0003),
    ),
})

print("\n=== Modo Artigo vs. Modo Engenharia — Casos 1, 2 e 3 do artigo ===")
comparar_modos({"Caso 1": caso1(), "Caso 2": caso2(), "Caso 3": caso3()})

print("\n=== Etilenoglicol 50%: diagnóstico (estrito=False) e solução com 4 passes ===")
eg = trocar_fluido_tubo(base, "etilenoglicol_50% (aprox. 30C)")
comparar(
    {"etilenoglicol (2 passes)": eg, "etilenoglicol (4 passes)": trocar_passes(eg, 4)},
    restricoes=RestricoesEngenharia(estrito=False),
)