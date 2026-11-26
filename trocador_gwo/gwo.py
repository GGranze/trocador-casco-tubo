"""Grey Wolf Optimizer — equações (33)-(39) do paper (Mirjalili et al., 2014).

Variantes de r1/r2 (eqs. 37-39):
  'classico'    : r ~ U(0,1)
  'exponencial' : r = 1/(1+exp(-Max_iter/1000))          (eq. 37, literal do paper)
  'gaussiana'   : r = 0,5 + 0,5·randn                    (eq. 38)
  'beta'        : r ~ Beta(a=0,1·U, b=U)                 (eq. 39 — INTERPRETAÇÃO:
                  o paper escreve beta(0.1rand, rand); assumi (a, b) = (0,1·U, U))
"""
import numpy as np


def _r(variante, rng, shape, max_iter):
    if variante == "classico":
        return rng.random(shape)
    if variante == "exponencial":
        return np.full(shape, 1.0 / (1.0 + np.exp(-max_iter / 1000.0)))
    if variante == "gaussiana":
        return 0.5 + 0.5 * rng.standard_normal(shape)
    if variante == "beta":
        a = np.maximum(0.1 * rng.random(shape), 1e-6)
        b = np.maximum(rng.random(shape), 1e-6)
        return rng.beta(a, b)
    raise ValueError("variante: classico | exponencial | gaussiana | beta")


def gwo(funcao, lb, ub, n_lobos=30, max_iter=200, variante="classico", seed=None,
        verbose=False):
    """Minimiza `funcao(x)` com x ∈ [lb, ub].

    Retorna (melhor_x, melhor_f, historico_do_melhor_por_iteracao).
    """
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    dim = lb.size

    X = lb + rng.random((n_lobos, dim)) * (ub - lb)
    f = np.array([funcao(x) for x in X])

    melhor_f, melhor_x = np.inf, X[0].copy()
    hist = []

    for t in range(max_iter):
        # alfa, beta, delta = 3 melhores da população atual
        idx = np.argsort(f)[:3]
        lideres = X[idx].copy()
        if f[idx[0]] < melhor_f:                       # elitismo
            melhor_f, melhor_x = float(f[idx[0]]), X[idx[0]].copy()
        hist.append(melhor_f)
        if verbose and t % 20 == 0:
            print(f"  it {t:4d}  melhor = {melhor_f:,.2f}")

        a = 2.0 * (1.0 - t / max_iter)                 # decai linearmente de 2 a 0
        soma = np.zeros_like(X)
        for lider in lideres:
            A = 2 * a * _r(variante, rng, X.shape, max_iter) - a    # eq. (35)
            C = 2 * _r(variante, rng, X.shape, max_iter)            # eq. (36)
            D = np.abs(C * lider - X)                               # eq. (33)
            soma += lider - A * D                                   # eq. (34)
        X = np.clip(soma / 3.0, lb, ub)                # X(t+1) = (X1+X2+X3)/3
        f = np.array([funcao(x) for x in X])

    i = int(np.argmin(f))                              # última geração também conta
    if f[i] < melhor_f:
        melhor_f, melhor_x = float(f[i]), X[i].copy()
    hist.append(melhor_f)
    return melhor_x, melhor_f, np.array(hist)
