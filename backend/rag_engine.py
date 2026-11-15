import os
import time
import json
import urllib.request
import urllib.error

ASTRONOMY_CORPUS = [
    {
        "id": "doc-01",
        "title": "Keplerian Orbital Mechanics & Perturbations",
        "doi": "10.1093/mnras/stx102",
        "content": "Kepler's laws describe ideal two-body motion: elliptical orbits with the central body at one focus, equal areas in equal times, and semi-major axis cubed proportional to orbital period squared. In Earth orbit, J2 oblateness causes nodal precession."
    },
    {
        "id": "doc-02",
        "title": "Low Earth Orbit (LEO) Space Debris Population",
        "doi": "10.1016/j.actaastro.2023.01.042",
        "content": "LEO debris between 200 km and 2000 km altitude poses collision hazards. The Kessler syndrome models cascading collisions creating self-sustaining debris fields. Tracking radar monitors fragments above 10 cm."
    },
    {
        "id": "doc-03",
        "title": "SDSS DR18 Photometric Classification of Stars, Galaxies, and Quasars",
        "doi": "10.3847/1538-4365/acb342",
        "content": "The Sloan Digital Sky Survey DR18 utilizes u, g, r, i, z photometric filters covering 300 to 1000 nm. Quasars exhibit strong ultraviolet excess and broad emission lines, while stars follow stellar locus sequences in color-color diagrams."
    },
    {
        "id": "doc-04",
        "title": "Spectroscopic Redshift in Extragalactic Astrophysics",
        "doi": "10.1086/133882",
        "content": "Cosmological redshift z = (lambda_obs - lambda_rest)/lambda_rest reflects the expansion of the universe. High redshift objects (z > 2) are observed in earlier cosmological epochs."
    },
    {
        "id": "doc-05",
        "title": "Solar Wind and Magnetospheric Space Weather",
        "doi": "10.1029/2022JA030514",
        "content": "Coronal mass ejections (CMEs) and solar flares detected by NASA DONKI induce geomagnetic storms. Geomagnetic induction can degrade satellite electronics and disrupt HF communications."
    },
    {
        "id": "doc-06",
        "title": "Planetary Atmospheres: Terrestrial vs Gas Giants",
        "doi": "10.1016/j.icarus.2022.115088",
        "content": "Venus features a runaway greenhouse atmosphere dominated by CO2 at 92 bar surface pressure. Jupiter and Saturn possess hydrogen-helium envelopes with atmospheric band circulation driven by internal heat and Coriolis forces."
    },
    {
        "id": "doc-07",
        "title": "Near-Earth Object (NEO) Hazard Assessment & Deflection",
        "doi": "10.1038/s41586-023-05810-5",
        "content": "NASA NeoWs continuously monitors Apollo, Amor, and Aten asteroid families. The DART mission demonstrated kinetic impactor deflection on Dimorphos, altering its orbital period around Didymos."
    },
    {
        "id": "doc-08",
        "title": "Exoplanet Transit Photometry and Transit Timing Variation",
        "doi": "10.3847/1538-3881/ab88ba",
        "content": "Periodic dips in stellar flux reveal transiting exoplanets. Transit timing variations (TTV) constrain multi-planet system architectures and planetary masses without radial velocity data."
    },
    {
        "id": "doc-09",
        "title": "Stellar Evolution and Supernova Progenitors",
        "doi": "10.1146/annurev-astro-081817-051803",
        "content": "Stars on the main sequence fuse hydrogen into helium. Massive stars (> 8 solar masses) undergo core collapse forming neutron stars or stellar-mass black holes, releasing type II supernovae."
    },
    {
        "id": "doc-10",
        "title": "Lagrangian Points and Orbital Equilibria in the Solar System",
        "doi": "10.1007/s10569-021-10023-x",
        "content": "The circular restricted three-body problem defines five equilibrium points. L1, L2, and L3 are collinear and unstable; L4 and L5 are triangular stable points hosting Trojan asteroids and space telescopes like JWST at Sun-Earth L2."
    },
    {
        "id": "doc-11",
        "title": "Mars Geology and Aqueous Alteration at Jezero Crater",
        "doi": "10.1126/science.abo5856",
        "content": "NASA's Perseverance rover at Jezero Crater has confirmed deltaic sedimentary deposits, olivine-rich igneous units, and carbonate alterations indicating a long-lived lacustrine environment 3.5 Ga ago."
    },
    {
        "id": "doc-12",
        "title": "Lunar Regolith Properties and Artemis Landing Site Characterization",
        "doi": "10.1016/j.pss.2023.105670",
        "content": "Permanently shadowed regions (PSRs) at the lunar south pole harbor water ice deposits. Low solar angles create extreme thermal regimes (-240 °C to +120 °C) critical for Artemis exploration."
    },
    {
        "id": "doc-13",
        "title": "Gravitational Waves and Compact Binary Coalescence",
        "doi": "10.1103/PhysRevLett.116.061102",
        "content": "LIGO and Virgo observe gravitational waves from binary black hole and neutron star mergers, confirming general relativistic quadrupole radiation and standard siren cosmography."
    },
    {
        "id": "doc-14",
        "title": "Cosmic Microwave Background and Lambda-CDM Cosmology",
        "doi": "10.1051/0004-6361/201833910",
        "content": "ESA Planck measurements of temperature anisotropies constrain baryon density, cold dark matter, and cosmological constant Lambda, estimating an age of 13.787 Gyr."
    },
    {
        "id": "doc-15",
        "title": "Black Hole Accretion Disks and Relativistic Jets",
        "doi": "10.1093/pasj/psz045",
        "content": "Magnetorotational instability (MRI) drives angular momentum transport in Shakura-Sunyaev disks. Relativistic Blandford-Znajek jets extract rotational energy from spinning Kerr black holes."
    },
    {
        "id": "doc-16",
        "title": "Heliospheric Boundary and Interstellar Medium Interaction",
        "doi": "10.1038/s41550-019-0928-3",
        "content": "Voyager 1 and 2 crossed the heliopause into interstellar space at ≈120 AU, measuring abrupt increases in cosmic ray density and magnetic field orientation changes."
    },
    {
        "id": "doc-17",
        "title": "Asteroid Spectral Taxonomy: C, S, and M Types",
        "doi": "10.1016/j.icarus.2020.113941",
        "content": "Asteroids are categorized into carbonaceous (C-type, low albedo, organic-rich), silicate (S-type, stony-iron), and metallic (M-type, nickel-iron, target of NASA Psyche mission)."
    },
    {
        "id": "doc-18",
        "title": "Dark Matter Halos and Galactic Rotation Curves",
        "doi": "10.1146/annurev-astro-091918-104445",
        "content": "Flat galactic rotation curves at large radii provide foundational evidence for non-baryonic dark matter halos dominating spiral and elliptical galaxy mass profiles."
    },
    {
        "id": "doc-19",
        "title": "James Webb Space Telescope Mid-Infrared Spectroscopy",
        "doi": "10.1038/s41586-022-05269-w",
        "content": "JWST MIRI and NIRSpec instruments analyze atmospheric transmission spectroscopy of transiting exoplanets, identifying CO2, H2O, SO2, and methane in WASP-39b."
    },
    {
        "id": "doc-20",
        "title": "Europa and Enceladus Subsurface Ocean Astrobiology",
        "doi": "10.1126/science.aai7842",
        "content": "Cassini INMS detection of molecular hydrogen (H2) and methane in Enceladus plume cryovolcanism indicates active hydrothermal vent systems at the ocean floor."
    }
]

class RAGEngine:
    def __init__(self):
        self.corpus = ASTRONOMY_CORPUS
        self.model = "gemini-3.5-flash"
        self.fallback_models = [
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest"
        ]

    @property
    def gemini_key(self):
        """Always read from env — no stale cached key after .env rotation."""
        return os.getenv("GEMINI_API_KEY", "")

    def hybrid_search(self, query: str, top_k: int = 3):
        terms = [t.lower() for t in query.split() if len(t) > 2]
        scored = []
        for doc in self.corpus:
            score = 0.0
            text = (doc["title"] + " " + doc["content"]).lower()
            for t in terms:
                # Term occurrences in body
                cnt = text.count(t)
                if cnt > 0:
                    score += 1.5 * min(cnt, 3)
                # Extra boost for title match
                if t in doc["title"].lower():
                    score += 3.5
            if score > 0:
                scored.append((score, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            return [d for _, d in scored[:top_k]]
        return self.corpus[:top_k]

    def ask(self, query: str, context_planet: str = None, mode: str = "standard", extra_context: str = None):
        t0 = time.time()
        contexts = self.hybrid_search(query, top_k=3)
        context_str = "\n".join([f"[{d['id']}] {d['title']}: {d['content']}" for d in contexts])

        role = "Orbit Overload's AI Space Intelligence Guide"
        tone = "Explain concepts clearly with scientific precision, concise formatting, and highlight key terms."
        if mode == "research":
            role = "Orbit Overload's Senior Astrophysical Research Fellow"
            tone = "Provide rigorous, quantitative astrophysical analysis, citing relevant physical laws, equations, and literature."

        planet_str = f"\nFocus Planet/Body: {context_planet}" if context_planet else ""
        extra_str = f"\nReal-Time Telemetry/Platform Context:\n{extra_context}" if extra_context else ""

        prompt = (
            f"Role: {role}.\n"
            f"Instructions: {tone}\n\n"
            f"Verified Scientific Literature Context:\n{context_str}{planet_str}{extra_str}\n\n"
            f"User Query: {query}\n\n"
            f"Please structure your response with clear scientific explanations. At the very end of your response, output exactly:\n"
            f"Follow-up Suggestions:\n"
            f"- [First relevant follow-up question]\n"
            f"- [Second relevant follow-up question]\n"
            f"- [Third relevant follow-up question]\n\n"
            f"Answer:"
        )

        raw_answer, error_msg = self._call_gemini(prompt)
        latency = round((time.time() - t0) * 1000, 2)

        # Parse follow-up suggestions
        answer = raw_answer
        suggested_followups = []
        if "Follow-up Suggestions:" in raw_answer:
            parts = raw_answer.split("Follow-up Suggestions:")
            answer = parts[0].strip()
            lines = [l.strip().lstrip("-*• 0123456789.)").strip() for l in parts[1].strip().splitlines() if l.strip()]
            suggested_followups = [l for l in lines if len(l) > 6][:3]

        if not suggested_followups:
            # Context-sensitive defaults
            if context_planet:
                suggested_followups = [
                    f"What is the atmospheric composition of {context_planet}?",
                    f"How does orbital mechanics affect {context_planet}'s seasonal cycle?",
                    f"What future space missions are targeted for {context_planet}?"
                ]
            else:
                suggested_followups = [
                    "How does this connect to current NASA mission discoveries?",
                    "What are the mathematical equations governing this phenomenon?",
                    "Can you generate an expert quiz to test my understanding?"
                ]

        if error_msg:
            source = "rag-local-fallback"
            answer = f"{answer} [Note: Gemini Live API unavailable: {error_msg}]"
        else:
            source = "rag-gemini"

        return {
            "query": query,
            "answer": answer,
            "suggested_followups": suggested_followups,
            "model": self.model,
            "latency_ms": latency,
            "source": source,
            "contexts": contexts,
            "mode": mode,
            "context_planet": context_planet
        }

    def research(self, query: str):
        t0 = time.time()
        contexts = self.hybrid_search(query, top_k=4)
        context_str = "\n".join([f"Source ({d['doi']}): {d['title']}\n{d['content']}" for d in contexts])

        prompt = (
            f"Generate a formal astrophysical research brief on: '{query}'.\n"
            f"Include:\n1. Abstract & Key Findings\n2. Observational Methodology\n"
            f"3. Spectral / Photometric Analysis\n4. Verified References & DOIs.\n\n"
            f"Context:\n{context_str}\n\nResearch Brief:"
        )

        brief, error_msg = self._call_gemini(prompt)
        latency = round((time.time() - t0) * 1000, 2)

        return {
            "query": query,
            "brief": brief if not error_msg else f"{brief} [Local RAG Synthesis, Live Gemini API: {error_msg}]",
            "doi_references": [d["doi"] for d in contexts],
            "citations": [d["title"] for d in contexts],
            "latency_ms": latency
        }

    def generate_report(self, topic: str, context_body: str = None):
        t0 = time.time()
        contexts = self.hybrid_search(topic, top_k=4)
        context_str = "\n".join([f"[{d['id']}] {d['title']} (DOI: {d['doi']}): {d['content']}" for d in contexts])

        body_str = f"Specific Focus: {context_body}\n" if context_body else ""
        prompt = (
            f"Generate a comprehensive, publication-ready Aerospace & Astrophysical Research Dossier on: '{topic}'.\n"
            f"{body_str}\n"
            f"Reference Data:\n{context_str}\n\n"
            f"Structure the dossier with Markdown formatting:\n"
            f"# RESEARCH DOSSIER: {topic.upper()}\n"
            f"## 1. Executive Summary & Abstract\n"
            f"## 2. Theoretical Framework & Mathematical Mechanics\n"
            f"## 3. Observational Data & Telescopic Instrumentation (NASA/ESA/SDSS)\n"
            f"## 4. Current Anomalies & Active Scientific Debates\n"
            f"## 5. Strategic Recommendations & Exploration Roadmaps\n"
            f"## 6. Verified References & Literature DOIs\n"
        )

        dossier, error_msg = self._call_gemini(prompt)
        latency = round((time.time() - t0) * 1000, 2)

        return {
            "topic": topic,
            "context_body": context_body,
            "dossier": dossier,
            "citations": [d["title"] for d in contexts],
            "dois": [d["doi"] for d in contexts],
            "latency_ms": latency,
            "model": self.model,
            "source": "rag-gemini" if not error_msg else "rag-local-fallback"
        }

    def _call_gemini(self, prompt: str) -> tuple[str, str]:
        """Returns (answer, error_message) with resilient multi-model fallback."""
        if not self.gemini_key or len(self.gemini_key) < 10:
            return (self._synthesize_rag_fallback(prompt), "No API key configured in .env")

        last_error = ""
        for model_name in self.fallback_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_key}"
            payload = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}]
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"}
            )

            try:
                with urllib.request.urlopen(req, timeout=12) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    text = res["candidates"][0]["content"]["parts"][0]["text"]
                    self.model = model_name
                    return (text, "")
            except urllib.error.HTTPError as e:
                err_detail = ""
                try:
                    raw = e.read().decode("utf-8")
                    parsed = json.loads(raw)
                    err_detail = parsed.get("error", {}).get("message", raw)
                except Exception:
                    err_detail = str(e)
                last_error = f"HTTP {e.code}: {err_detail}"
                continue
            except Exception as e:
                last_error = str(e)
                continue

        return (self._synthesize_rag_fallback(prompt), last_error)

    def _synthesize_rag_fallback(self, prompt: str) -> str:
        # Extract query portion
        q = prompt.split("User Query: ")[-1].split("\n\nAnswer:")[0] if "User Query: " in prompt else ""
        contexts = self.hybrid_search(q, top_k=2)
        summary = " ".join([f"{c['title']}: {c['content']}" for c in contexts])
        return f"Orbit Overload RAG Knowledge Synthesis: {summary}"

rag_engine = RAGEngine()
