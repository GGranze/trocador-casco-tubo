"""Compara o modelo com os valores publicados nas Tabelas 5, 6 e 7.

    python -m trocador_gwo.validar
"""
from .casos import CASOS, PROJETOS_PAPER, REFERENCIA_PAPER
from .modelo import avaliar

CAMPOS = ["Nt", "vt", "ht", "hs", "U", "A", "L", "Ci", "Co", "Cod", "Ctot"]


def main():
    for n, fabrica in CASOS.items():
        esp = fabrica()
        for nome, (Ds, do, B) in PROJETOS_PAPER[n].items():
            r = avaliar(esp, Ds, do, B)
            calc = dict(Nt=r.Nt, vt=r.v_t, ht=r.h_t, hs=r.h_s, U=r.U, A=r.A, L=r.L,
                        Ci=r.Ci, Co=r.Co, Cod=r.Cod, Ctot=r.Ctot)
            ref = REFERENCIA_PAPER[(n, nome)]
            print(f"\nCaso {n} — {nome}  (Ds={Ds}, do={do}, B={B})")
            print(f"  {'':5s}{'paper':>12s}{'modelo':>12s}{'erro %':>9s}")
            for c in CAMPOS:
                e = 100 * (calc[c] - ref[c]) / ref[c]
                print(f"  {c:5s}{ref[c]:12.4g}{calc[c]:12.4g}{e:9.1f}")


if __name__ == "__main__":
    main()
