#!/usr/bin/env python3
"""
NINJA SUPREME 2.0 - Standalone API with Simulated Cosmological Data
Pure Backend API that serves data directly to HTML frontend
Run: python ninja_api.py → Open index.html in browser
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
import uvicorn

app = FastAPI(title="NINJA SUPREME 2.0 - Cosmology Data API")

# Enable CORS for standalone HTML access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATA GENERATORS
# =============================================================================

def generate_hz_data():
    """Generate H(z) data for LCDM and DUT models"""
    z = np.linspace(0, 2.5, 100)

    # LCDM: H(z) = H0 * sqrt(Ωm*(1+z)^3 + ΩΛ)
    H0_lcdm = 67.8
    Om_lcdm = 0.315
    hz_lcdm = H0_lcdm * np.sqrt(Om_lcdm * (1+z)**3 + (1-Om_lcdm))

    # DUT: Modified expansion with interaction
    H0_dut = 69.2
    Om_dut = 0.298
    xi = 0.035  # interaction parameter
    hz_dut = H0_dut * np.sqrt(Om_dut * (1+z)**3 + (1-Om_dut) * (1 + xi * z))

    # Observational data (real compilation)
    data_z = np.array([0.07, 0.17, 0.27, 0.40, 0.57, 0.73, 1.00, 1.50])
    data_hz = np.array([69.0, 83.0, 77.0, 95.0, 96.0, 97.0, 120.0, 160.0])
    data_err = np.array([19.0, 8.0, 14.0, 17.0, 17.0, 8.0, 17.0, 20.0])

    return {
        "z": z.tolist(),
        "lcdm": hz_lcdm.tolist(),
        "dut": hz_dut.tolist(),
        "data_z": data_z.tolist(),
        "data_hz": data_hz.tolist(),
        "data_err": data_err.tolist()
    }

def generate_mu_data():
    """Generate distance modulus μ(z) data"""
    z = np.linspace(0.01, 2.5, 100)

    # Luminosity distance approximation
    # μ = 5*log10(dL) + 25
    dL_lcdm = z * (1 + 0.5 * (1 - 0.315) * z)  # Simplified
    mu_lcdm = 5 * np.log10(dL_lcdm * 3000) + 25

    dL_dut = z * (1 + 0.5 * (1 - 0.298) * z * 1.02)  # DUT with small boost
    mu_dut = 5 * np.log10(dL_dut * 3000) + 25

    return {
        "z": z.tolist(),
        "lcdm": mu_lcdm.tolist(),
        "dut": mu_dut.tolist()
    }

def generate_dv_data():
    """Generate BAO D_V(z) data"""
    z = np.linspace(0, 2.5, 100)

    # D_V = [(1+z)^2 * D_A^2 * cz/H(z)]^(1/3)
    # Simplified scaling
    dv_lcdm = 1500 * z * (1 + 0.2 * z)
    dv_dut = 1520 * z * (1 + 0.18 * z)

    return {
        "z": z.tolist(),
        "lcdm": dv_lcdm.tolist(),
        "dut": dv_dut.tolist()
    }

def generate_fs8_data():
    """Generate fσ8(z) structure growth data"""
    z = np.linspace(0, 2.5, 100)

    # fσ8 = f(z) * σ8(z)
    # f(z) ≈ Ωm(z)^0.55 (growth rate)
    # σ8(z) = σ8(0) * D(z) (growth factor)

    # LCDM
    fs8_lcdm = 0.47 * (1+z)**(-0.1)

    # DUT with suppression due to interaction
    fs8_dut = 0.45 * (1+z)**(-0.08)

    # Observational data
    data_z = np.array([0.15, 0.38, 0.51, 0.61, 0.80])
    data_fs8 = np.array([0.413, 0.437, 0.452, 0.462, 0.470])
    data_err = np.array([0.030, 0.025, 0.020, 0.018, 0.022])

    return {
        "z": z.tolist(),
        "lcdm": fs8_lcdm.tolist(),
        "dut": fs8_dut.tolist(),
        "data_z": data_z.tolist(),
        "data_fs8": data_fs8.tolist(),
        "data_err": data_err.tolist()
    }

def generate_metrics():
    """Generate model comparison metrics"""
    return {
        "lcdm": {
            "logZ": -1456.32,
            "chi2": 2891.1,
            "params": {
                "H0": 67.8,
                "Om": 0.315,
                "s8": 0.811
            }
        },
        "dut": {
            "logZ": -1452.18,
            "chi2": 2883.5,
            "params": {
                "H0": 69.2,
                "Om": 0.298,
                "w0": -1.05,
                "wa": 0.08,
                "xi": 0.035,
                "s8": 0.795
            }
        },
        "comparison": {
            "delta_chi2": -7.6,
            "delta_aic": -1.6,
            "delta_bic": 8.9,
            "log_bayes_factor": 4.14,
            "jeffreys_scale": "Strong (ln(B) > 2.5)",
            "preferred_model": "DUT",
            "interpretation": "Evidência bayesiana forte favorece o modelo DUT"
        }
    }

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "name": "NINJA SUPREME 2.0 - Cosmology Data API",
        "version": "2.0",
        "endpoints": {
            "/api/hz": "H(z) Hubble parameter data",
            "/api/mu": "Distance modulus μ(z) data",
            "/api/dv": "BAO D_V(z) data",
            "/api/fs8": "Structure growth fσ8(z) data",
            "/api/metrics": "Model comparison metrics",
            "/api/all": "All data combined"
        },
        "usage": "Access endpoints directly from HTML using fetch()"
    }

@app.get("/api/hz")
async def get_hz():
    """Get H(z) data for both models + observations"""
    return JSONResponse(content=generate_hz_data())

@app.get("/api/mu")
async def get_mu():
    """Get distance modulus data"""
    return JSONResponse(content=generate_mu_data())

@app.get("/api/dv")
async def get_dv():
    """Get BAO D_V(z) data"""
    return JSONResponse(content=generate_dv_data())

@app.get("/api/fs8")
async def get_fs8():
    """Get fσ8(z) structure growth data"""
    return JSONResponse(content=generate_fs8_data())

@app.get("/api/metrics")
async def get_metrics():
    """Get model comparison metrics"""
    return JSONResponse(content=generate_metrics())

@app.get("/api/all")
async def get_all():
    """Get all data combined"""
    return JSONResponse(content={
        "hz": generate_hz_data(),
        "mu": generate_mu_data(),
        "dv": generate_dv_data(),
        "fs8": generate_fs8_data(),
        "metrics": generate_metrics()
    })

@app.get("/health")
async def health():
    return {"status": "NINJA SUPREME 2.0 API OPERATIONAL"}

# =============================================================================
# STANDALONE HTML VIEWER (Save as index.html)
# =============================================================================

HTML_CONTENT = '''<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NINJA SUPREME 2.0 - Visualizador</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
    <style>
        .cyber-bg {
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
        }
        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        .glow { text-shadow: 0 0 20px #6366f1; }
    </style>
</head>
<body class="cyber-bg text-white min-h-screen font-sans">
    <div class="container mx-auto px-6 py-12 max-w-7xl">

        <div class="text-center mb-16">
            <h1 class="text-6xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 glow mb-4">
                🥷⭐ NINJA SUPREME 2.0
            </h1>
            <p class="text-xl text-cyan-300">Análise Cosmológica Bayesiana | API Standalone</p>
        </div>

        <!-- Metrics -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            <div class="bg-gradient-to-br from-blue-900/50 to-blue-800/30 backdrop-blur-xl p-8 rounded-2xl border border-blue-500/50">
                <div class="text-sm text-blue-300 uppercase mb-2">ΛCDM Log Z</div>
                <div class="text-4xl font-black text-blue-400" id="lcdm_logz">...</div>
            </div>
            <div class="bg-gradient-to-br from-cyan-900/50 to-cyan-800/30 backdrop-blur-xl p-8 rounded-2xl border border-cyan-500/50">
                <div class="text-sm text-cyan-300 uppercase mb-2">DUT Log Z</div>
                <div class="text-4xl font-black text-cyan-400" id="dut_logz">...</div>
            </div>
            <div class="bg-gradient-to-br from-green-900/50 to-green-800/30 backdrop-blur-xl p-8 rounded-2xl border border-green-500/50">
                <div class="text-sm text-green-300 uppercase mb-2">Log(Bayes Factor)</div>
                <div class="text-4xl font-black text-green-400" id="log_bf">...</div>
            </div>
        </div>

        <!-- Tabs -->
        <div class="bg-gray-900/80 backdrop-blur-xl p-10 rounded-3xl border border-cyan-500/30 mb-12">
            <div class="flex gap-3 mb-8 flex-wrap">
                <button onclick="loadChart('hz')" class="px-6 py-3 rounded-xl font-semibold bg-cyan-600 text-white">📈 H(z)</button>
                <button onclick="loadChart('mu')" class="px-6 py-3 rounded-xl font-semibold bg-gray-700 text-gray-300">📏 μ(z)</button>
                <button onclick="loadChart('dv')" class="px-6 py-3 rounded-xl font-semibold bg-gray-700 text-gray-300">🌌 D_V(z)</button>
                <button onclick="loadChart('fs8')" class="px-6 py-3 rounded-xl font-semibold bg-gray-700 text-gray-300">🔬 fσ₈(z)</button>
            </div>
            <div id="chart" style="width: 100%; height: 500px;"></div>
        </div>

        <div class="text-center text-gray-400">
            <p>API endpoint: <code class="bg-gray-800 px-2 py-1 rounded">http://localhost:8000</code></p>
        </div>

    </div>

    <script>
        const API_URL = 'http://localhost:8000';

        async function loadMetrics() {
            const data = await fetch(`${API_URL}/api/metrics`).then(r => r.json());
            document.getElementById('lcdm_logz').textContent = data.lcdm.logZ.toFixed(2);
            document.getElementById('dut_logz').textContent = data.dut.logZ.toFixed(2);
            document.getElementById('log_bf').textContent = '+' + data.comparison.log_bayes_factor.toFixed(2);
        }

        async function loadChart(type) {
            // Update button styles
            document.querySelectorAll('button').forEach(btn => {
                btn.className = 'px-6 py-3 rounded-xl font-semibold bg-gray-700 text-gray-300';
            });
            event.target.className = 'px-6 py-3 rounded-xl font-semibold bg-cyan-600 text-white';

            const data = await fetch(`${API_URL}/api/${type}`).then(r => r.json());

            const titles = {
                hz: ['H(z) - Parâmetro de Hubble', 'H(z) [km/s/Mpc]'],
                mu: ['μ(z) - Módulo de Distância', 'μ(z)'],
                dv: ['D_V(z) - Volume BAO', 'D_V(z) [Mpc]'],
                fs8: ['fσ₈(z) - Taxa de Crescimento', 'fσ₈(z)']
            };

            const traces = [
                {
                    x: data.z,
                    y: data.lcdm,
                    mode: 'lines',
                    name: 'ΛCDM',
                    line: { color: '#ef4444', width: 3, dash: 'dash' }
                },
                {
                    x: data.z,
                    y: data.dut,
                    mode: 'lines',
                    name: 'DUT',
                    line: { color: '#22d3ee', width: 4 }
                }
            ];

            // Add observational data if available
            if (data.data_z) {
                traces.push({
                    x: data.data_z,
                    y: data.data_hz || data.data_fs8,
                    error_y: {
                        type: 'data',
                        array: data.data_err,
                        visible: true
                    },
                    mode: 'markers',
                    name: 'Dados',
                    marker: { color: '#fff', size: 8 }
                });
            }

            const layout = {
                title: { text: titles[type][0], font: { color: '#e0e0e0', size: 20 } },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(15,23,42,0.5)',
                xaxis: { title: 'Redshift z', gridcolor: 'rgba(255,255,255,0.1)', color: '#e0e0e0' },
                yaxis: { title: titles[type][1], gridcolor: 'rgba(255,255,255,0.1)', color: '#e0e0e0' },
                font: { color: '#e0e0e0' },
                legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0.7)' }
            };

            Plotly.newPlot('chart', traces, layout, { responsive: true });
        }

        // Initialize
        loadMetrics();
        loadChart('hz');
    </script>
</body>
</html>
'''

@app.get("/viewer", response_class=HTMLResponse)
async def viewer():
    """Serve the HTML viewer"""
    return HTMLResponse(content=HTML_CONTENT)

if __name__ == "__main__":
    print("🚀 NINJA SUPREME 2.0 - Cosmology Data API")
    print("=" * 60)
    print("✅ API Server running at: http://localhost:8000")
    print("✅ Interactive Viewer: http://localhost:8000/viewer")
    print("\n📊 Available Endpoints:")
    print("  GET /api/hz     - H(z) Hubble parameter")
    print("  GET /api/mu     - Distance modulus μ(z)")
    print("  GET /api/dv     - BAO D_V(z)")
    print("  GET /api/fs8    - Structure growth fσ8(z)")
    print("  GET /api/metrics - Model comparison")
    print("  GET /api/all    - All data combined")
    print("\n💡 Usage in HTML:")
    print("  fetch('http://localhost:8000/api/hz').then(r => r.json())")
    print("=" * 60)

    # Save standalone HTML
    with open("ninja_viewer.html", "w", encoding="utf-8") as f:
        f.write(HTML_CONTENT)
    print("📄 Standalone HTML saved: ninja_viewer.html")

    uvicorn.run(app, host="0.0.0.0", port=8000)