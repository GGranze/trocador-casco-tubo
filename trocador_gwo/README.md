# trocador_gwo — otimização casco-tubo por GWO (Vieira, Mariani & Rieger)

Rodar:  `python -m trocador_gwo.exemplo`   |   `python -m trocador_gwo.validar`
Requer apenas `numpy`. Execute a partir da pasta que CONTÉM `trocador_gwo/`.

| módulo        | conteúdo (equações do paper)                                   |
|---------------|-----------------------------------------------------------------|
| fluidos.py    | Tabelas 2-4 + fluidos extras (aprox., fora do paper) + `fluido_custom` |
| geometria.py  | (15)-(20), (23)/Tabela 1, L = A/(π·do·Nt)                       |
| termica.py    | (1)-(14), (21), (22): LMTD, F, h tubo/casco, U, área            |
| hidraulica.py | (28)-(32): ΔP, potência de bombeamento                          |
| custos.py     | (24)-(27)                                                       |
| modelo.py     | `Especificacao` (entradas) + `avaliar(esp, Ds, do, B)`          |
| gwo.py        | (33)-(39): GWO clássico, exponencial, gaussiana, beta           |
| otimizador.py | liga GWO ao modelo (variáveis Ds, do, B; minimiza Ctot)         |
| cenarios.py   | trocar fluido/passes/arranjo/temperaturas/carga + `comparar()`  |
| casos.py      | casos 1-3 e projetos publicados (Tabelas 5-7)                   |

## Desvios e ambiguidades do paper (decisões tomadas)
1. a3 = 0,91 (o texto diz 0,77; só 0,91 reproduz o Ci das tabelas, erro ≈0,01 %).
2. Limites de do e B estão trocados no texto; usei do∈[15;51] mm, B∈[0,05;0,5] m.
3. Eq. (6) omite a condução na parede (o texto a cita): segui a eq.; `k_parede` opcional.
4. Razão de viscosidade (μ/μw)^0,14 = 1 (o paper não dá μw). Informe `mu_parede` no Fluido.
5. Eq. (37) exponencial e (39) beta são ambíguas; implementadas literalmente/interpretadas (ver gwo.py).
6. F (eq. 5) só vale para 1 passe no casco; F=1 para 1 passe nos tubos.
7. Nº de passes limitado a 1, 2, 4, 6, 8 (Tabela 1).

## Fidelidade numérica
As Tabelas 5-7 NÃO são reproduzidas exatamente (ver `validar.py`): as colunas "original"
vêm da literatura (Nt real de projeto, não eq. 23) e há inconsistências internas no
paper (ex.: caso 1/GWO: A=179,3 m² não satisfaz q=U·A·F·ΔTml com o U publicado).
