# Computational Materials Science

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Weekly programming assignments for a course in computational materials science. Each week a numerical method is written as tested code and shown to be correct against theory (**verification**: exact error formulas, closed forms, exact discrete solutions, high-precision reruns), then applied to a materials problem and compared with experiment or literature data (**validation**).

Python 3 with NumPy, SciPy and Matplotlib. Every core algorithm is written from scratch; library routines are used only for comparison.

## Weeks

| Week | Topic | Key results |
|---|---|---|
| [week01](week01/) | Convergence of numerical quadrature; Debye heat capacity of Cu | Observed orders 2.000 / 4.000 for the trapezoid / Simpson rules; the error for √x matches its exact error formula to a relative 10⁻⁹, and for e^{cos x} down to round-off; with the electronic heat capacity included, C_p of Cu at 298 K is only −0.6% from experiment |
| [week02](week02/) | The V-shaped error of finite differences and the optimal step; bulk modulus of Cu | Every finite-difference error equals the exact truncation series plus round-off, and the round-off stays inside the textbook bound; the complex step has exactly zero error for h ≤ 10⁻⁸; the optimal step for the Debye U(T), 3×10⁻⁴ K, is set by its characteristic scale of 63.5 K; B of Cu is 139.63 GPa, and with 10⁻⁵ eV of noise in the energy the best strain is about 1.2% |

Each week's README lists the tasks, the key results, a validation table (every value next to its independent check) and where the results differ from the course's reference solution.

## Repository layout

```text
Computational-Materials-Science/
├── environment.yml          # conda environment "cms"
└── weekNN/                  # one folder per week, all with the same structure
    ├── README.md            # tasks, key results, validation table, figures
    └── work/
        ├── *.py             # modules
        ├── test_*.py        # pytest suite
        ├── labNN.ipynb      # executed notebook: every number and figure
        ├── labNN.html       # the same notebook as a web page
        └── figures/
```

Later weeks import earlier weeks' code (week02 reuses the Simpson rule and Debye C_V of week01), so clone the whole repository.

## Quick start

```bash
git clone https://github.com/Shuruyue/Computational-Materials-Science.git
cd Computational-Materials-Science
conda env create -f environment.yml   # once
conda activate cms
cd week01/work                        # or any other week
pytest -q                             # automated tests
jupyter lab lab01.ipynb               # Run All reproduces every number and figure
```

To read the results without running anything, open `lab*.ipynb` on GitHub (outputs included), or download `lab*.html` and open it in a browser.

## Tests

| Week | Tests | What they check besides the assignment's acceptance criteria |
|---|---|---|
| week01 | 40 | exact error formulas for sin x, √x and e^{cos x}; Debye limits up to u⁶ |
| week02 | 88 | every finite-difference error against its exact truncation series and the round-off bound; series solutions of the Debye U(T) |

## Not included

The course handouts, textbook page lists, assignment statements, reference solutions and starter code, as well as the textbooks and papers, stay local. The notebooks sometimes compare with the course's reference solution, called "05"; it is not part of this repository.

## License

[MIT](LICENSE)
