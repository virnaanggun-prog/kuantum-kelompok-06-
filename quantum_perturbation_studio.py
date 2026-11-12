import math
import re
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
from PIL import Image
from scipy.integrate import quad
from scipy.special import eval_hermite, factorial

# ============================================================
# QUANTUM PERTURBATION STUDIO
# First-order stationary perturbation theory
#
# Models:
# 1. Infinite square well: 0 < x < a
# 2. Harmonic oscillator: dimensionless units hbar = m = omega = 1
# ============================================================

st.set_page_config(
    page_title="Quantum Perturbation Studio",
    page_icon="⚛️",
    layout="wide",
)

# ---------- Theme ----------
BG = "#F3F6FB"
NAVY = "#14213D"
BLUE = "#2563EB"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {BG}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .main-header {{
        background: {NAVY}; color: white; padding: 24px 28px;
        border-radius: 16px; margin-bottom: 18px;
    }}
    .main-header h1 {{ margin: 0; font-size: 30px; }}
    .main-header p {{ color: #D8E2F0; margin: 6px 0 0 0; }}
    .section {{
        background: white; border: 1px solid #E2E8F0;
        border-radius: 14px; padding: 18px; margin-bottom: 14px;
    }}
    div.stButton > button {{
        min-height: 42px; border-radius: 9px; font-weight: 650;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
      <h1>⚛️ Quantum Perturbation Studio</h1>
      <p>Kalkulator teori perturbasi tak bergantung waktu · Energi dan fungsi gelombang orde pertama</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Symbols and safe expression parser ----------
x = sp.Symbol("x", real=True)
lam_symbol = sp.Symbol("lam", real=True)
V1_symbol = sp.Symbol("V1", real=True)
a_symbol = sp.Symbol("a", positive=True, real=True)

ALLOWED_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "sqrt": sp.sqrt,
    "log": sp.log,
    "abs": sp.Abs,
}
ALLOWED_NAMES = {
    "x", "lam", "V1", "a", "pi",
    *ALLOWED_FUNCTIONS.keys(),
}


def normalize_expression(raw: str) -> str:
    """Convert common handwritten/typed math notation into Python notation."""
    text = raw.strip()
    text = re.sub(r"^\s*H\s*'\s*\(\s*x\s*\)\s*=", "", text, flags=re.I)
    text = re.sub(r"^\s*H\s*'\s*=", "", text, flags=re.I)
    text = text.replace("−", "-").replace("×", "*").replace("·", "*")
    text = text.replace("÷", "/").replace("π", "pi")
    text = text.replace("λ", "lam").replace("Λ", "lam")
    text = text.replace("³", "**3").replace("²", "**2")
    text = text.replace("⁴", "**4").replace("⁵", "**5")
    text = text.replace("^", "**")
    text = re.sub(r"\blambda\b", "lam", text, flags=re.I)

    # Make common implicit products explicit: lamx -> lam*x, 2x -> 2*x.
    text = re.sub(r"lam\s*(?=x\b)", "lam*", text)
    text = re.sub(r"\bV1\s*(?=x\b)", "V1*", text)
    text = re.sub(r"(?<=\d)\s*(?=x\b)", "*", text)
    text = re.sub(r"(?<=\))\s*(?=x\b)", "*", text)
    text = re.sub(r"\s+", "", text)
    return text


def parse_perturbation(raw: str, lam_value: float, v1_value: float, a_value: float):
    expression = normalize_expression(raw)
    if not expression:
        raise ValueError("Fungsi perturbasi belum diisi.")
    if len(expression) > 160:
        raise ValueError("Ekspresi terlalu panjang (maksimal 160 karakter).")

    # Only permit mathematical characters and known names before sympify.
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/().,]*", expression):
        raise ValueError(
            "Ada karakter yang belum didukung. Gunakan x, λ/lam, V1, a, "
            "angka, operasi + - * / **, dan fungsi matematika yang tersedia."
        )

    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression))
    unknown = names - ALLOWED_NAMES
    if unknown:
        raise ValueError("Simbol/fungsi tidak dikenal: " + ", ".join(sorted(unknown)))

    local_dict = {
        "x": x,
        "lam": sp.Float(lam_value),
        "V1": sp.Float(v1_value),
        "a": sp.Float(a_value),
        "pi": sp.pi,
        **ALLOWED_FUNCTIONS,
    }
    try:
        expr = sp.sympify(expression, locals=local_dict, evaluate=True)
    except Exception as exc:
        raise ValueError(
            "Format fungsi belum dapat dibaca. Contoh yang benar: lam*x**3 atau V1*x/a."
        ) from exc

    if expr.free_symbols - {x}:
        raise ValueError("Ekspresi hanya boleh bergantung pada variabel x.")
    if expr.has(sp.Derivative, sp.Integral, sp.Sum):
        raise ValueError("Operator tersebut belum didukung.")
    return expr, expression


# ---------- Basis functions and matrix elements ----------
def psi_well(q: int, xx: float, width: float) -> float:
    return math.sqrt(2.0 / width) * math.sin(q * math.pi * xx / width)


def psi_ho(q: int, xx: float) -> float:
    """Harmonic-oscillator eigenfunction for hbar=m=omega=1; q starts at 0."""
    norm = 1.0 / math.sqrt((2.0 ** q) * factorial(q) * math.sqrt(math.pi))
    return float(norm * eval_hermite(q, xx) * np.exp(-(xx ** 2) / 2.0))


def matrix_element_well(k: int, n: int, expr, width: float) -> float:
    f = sp.lambdify(x, expr, modules=["numpy"])

    def integrand(xx):
        return psi_well(k, xx, width) * float(f(xx)) * psi_well(n, xx, width)

    value, _ = quad(integrand, 0.0, width, epsabs=1e-10, epsrel=1e-10, limit=250)
    return float(value)


def matrix_element_ho(k: int, n: int, expr) -> float:
    f = sp.lambdify(x, expr, modules=["numpy"])

    def integrand(xx):
        return psi_ho(k, xx) * float(f(xx)) * psi_ho(n, xx)

    value, _ = quad(
        integrand, -np.inf, np.inf,
        epsabs=1e-10, epsrel=1e-10, limit=300
    )
    return float(value)


# ---------- Sidebar / inputs ----------
with st.sidebar:
    st.header("Pengaturan model")
    model = st.selectbox(
        "Pilih sistem kuantum",
        ["Osilator harmonik", "Sumur potensial tak hingga"],
    )
    st.caption(
        "Osilator harmonik menggunakan satuan tak berdimensi "
        "ℏ = m = ω = 1. Sumur tak hingga menggunakan 0 < x < a."
    )

left, right = st.columns([5, 6], gap="large")

with left:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("01 | Input perhitungan")

    default_expression = "λx³" if model == "Osilator harmonik" else "V1*x/a"
    expression = st.text_input(
        "Fungsi perturbasi H′(x)",
        value=default_expression,
        help="Contoh osilator: λx³ atau lam*x**3. Contoh sumur: V1*x/a.",
    )

    uploaded_image = st.file_uploader(
        "Upload gambar fungsi (opsional, PNG/JPG)",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            st.image(image, caption="Gambar fungsi yang diunggah", use_container_width=True)
            st.info(
                "Gambar ditampilkan sebagai referensi. Ketik ulang fungsi pada kolom di atas "
                "dan periksa pangkat serta simbolnya sebelum menghitung."
            )
        except Exception:
            st.warning("File gambar tidak dapat dibuka.")

    st.markdown("**Parameter sistem**")
    if model == "Osilator harmonik":
        lam_value = st.number_input("Konstanta perturbasi λ", value=0.1, step=0.05, format="%.5f")
        n_input = st.number_input(
            "Bilangan kuantum n (dimulai dari 0)",
            min_value=0, max_value=20, value=0, step=1,
        )
        state_limit = st.number_input(
            "Jumlah keadaan basis yang dihitung",
            min_value=2, max_value=30, value=10, step=1,
        )
        st.caption("Energi tak terganggu: Eₙ⁽⁰⁾ = n + ½ (dalam satuan ℏω = 1).")
        a_value, v1_value = 1.0, 1.0
    else:
        col1, col2 = st.columns(2)
        with col1:
            a_value = st.number_input("Lebar sumur a", min_value=0.000001, value=1.0, step=0.1)
            e1_value = st.number_input("Energi dasar E₁⁽⁰⁾", min_value=0.000001, value=1.0, step=0.1)
            n_input = st.number_input("Bilangan kuantum n (dimulai dari 1)", min_value=1, max_value=30, value=1, step=1)
        with col2:
            v1_value = st.number_input("Parameter potensial V1", value=1.0, step=0.1)
            state_limit = st.number_input("Batas keadaan N", min_value=2, max_value=30, value=8, step=1)
        lam_value = 1.0
        st.caption("Energi tak terganggu: Eₙ⁽⁰⁾ = n² E₁⁽⁰⁾.")

    calculate = st.button("Hitung", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("02 | Output hasil")
    st.write("Hasil perhitungan akan tampil di sini setelah tombol **Hitung** ditekan.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Calculation ----------
if calculate:
    try:
        if model == "Osilator harmonik" and n_input >= state_limit:
            raise ValueError("Jumlah keadaan basis harus lebih besar daripada indeks n.")
        if model == "Sumur potensial tak hingga" and n_input > state_limit:
            raise ValueError("Batas keadaan N harus lebih besar atau sama dengan n.")

        expr, normalized = parse_perturbation(
            expression, float(lam_value), float(v1_value), float(a_value)
        )

        if model == "Osilator harmonik":
            n = int(n_input)
            N = int(state_limit)
            h_nn = matrix_element_ho(n, n, expr)
            e0 = n + 0.5
            energy_correction = h_nn
            energy_total = e0 + energy_correction
            matrix_element = matrix_element_ho
            basis_label = "n (indeks mulai 0)"
            states = range(N)
            denominator_fn = lambda m: (n + 0.5) - (m + 0.5)
            psi_label = f"ψ_{n}(x)"
        else:
            n = int(n_input)
            N = int(state_limit)
            h_nn = matrix_element_well(n, n, expr, float(a_value))
            e0 = (n ** 2) * float(e1_value)
            energy_correction = h_nn
            energy_total = e0 + energy_correction
            matrix_element = lambda k, q, ex: matrix_element_well(k, q, ex, float(a_value))
            basis_label = "m (indeks mulai 1)"
            states = range(1, N + 1)
            denominator_fn = lambda m: ((n ** 2) - (m ** 2)) * float(e1_value)
            psi_label = f"ψ_{n}(x)"

        rows = []
        wave_terms = [psi_label]
        for m in states:
            if m == n:
                continue
            h_mn = matrix_element(m, n, expr)
            denominator = float(denominator_fn(m))
            coefficient = h_mn / denominator
            rows.append({
                basis_label: m,
                "⟨m|H′|n⟩": h_mn,
                "Eₙ⁽⁰⁾ − Eₘ⁽⁰⁾": denominator,
                "cₘₙ⁽¹⁾": coefficient,
            })
            if abs(coefficient) > 1e-12:
                wave_terms.append(f"({coefficient:.6g}) ψ_{m}(x)")

        with right:
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.subheader("Hasil koreksi energi")
            st.write(f"**Model:** {model}")
            st.write(f"**Fungsi yang dibaca:** `{sp.sstr(expr)}`")
            st.write(f"**Notasi hasil normalisasi:** `{normalized}`")
            st.metric("Energi tak terganggu Eₙ⁽⁰⁾", f"{e0:.10g}")
            st.metric("Koreksi energi orde pertama Eₙ⁽¹⁾", f"{energy_correction:.10g}")
            st.metric("Energi hingga orde pertama", f"{energy_total:.10g}")
            if model == "Osilator harmonik" and sp.simplify(expr.subs(x, -x) + expr) == 0:
                st.info(
                    "Untuk perturbasi ganjil λx³, koreksi energi orde pertama bernilai nol "
                    "pada setiap keadaan osilator harmonik karena simetri/paritas."
                )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.subheader("Koefisien koreksi fungsi gelombang")
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.write("Tidak ada keadaan lain pada rentang basis yang dipilih.")
            st.markdown("**Pendekatan fungsi gelombang terkoreksi**")
            st.code("ψₙ⁽¹⁾ ≈ " + " + ".join(wave_terms))
            st.caption(
                "Koefisien dihitung dengan cₘₙ⁽¹⁾ = ⟨m|H′|n⟩ / "
                "(Eₙ⁽⁰⁾ − Eₘ⁽⁰⁾). Hasil fungsi gelombang merupakan pendekatan "
                "terpotong sesuai jumlah keadaan basis yang dipilih."
            )
            st.success("Perhitungan selesai.")
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as exc:
        with right:
            st.error(f"Input/perhitungan tidak valid: {exc}")

st.caption(
    "Quantum Perturbation Studio · Hasil numerik bergantung pada model, satuan, "
    "fungsi perturbasi, dan jumlah keadaan basis."
)
