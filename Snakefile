rule rk_example:
    output:
        "src/tex/output/mc_out.txt"
        "src/tex/output/rk_out.txt"
        "src/tex/output/rk_err.txt"
    script:
        "src/scripts/compare_massive.py"