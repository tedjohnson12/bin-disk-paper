rule rk_example:
    output:
        "src/tex/output/mc_out.txt"
        "src/tex/output/rk_out.txt"
        "src/tex/output/rk_err.txt"
    script:
        "src/scripts/compare_massive.py"

rule high_j:
    output:
        "src/tex/output/high_j_integral.txt"
    script:
        "src/scripts/high_j_integral.py"