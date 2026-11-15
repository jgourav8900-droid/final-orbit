from fastapi import APIRouter

router = APIRouter(prefix="/api/planets", tags=["Planets"])

PLANET_CATALOGUE = [
    {"name": "Sun", "type": "Star", "diameter_km": 1392700, "distance_from_sun_au": 0.0, "orbital_period_days": 0.0, "moons": 0},
    {"name": "Mercury", "type": "Terrestrial", "diameter_km": 4879, "distance_from_sun_au": 0.39, "orbital_period_days": 88.0, "moons": 0},
    {"name": "Venus", "type": "Terrestrial", "diameter_km": 12104, "distance_from_sun_au": 0.72, "orbital_period_days": 224.7, "moons": 0},
    {"name": "Earth", "type": "Terrestrial", "diameter_km": 12742, "distance_from_sun_au": 1.00, "orbital_period_days": 365.25, "moons": 1},
    {"name": "Mars", "type": "Terrestrial", "diameter_km": 6779, "distance_from_sun_au": 1.52, "orbital_period_days": 687.0, "moons": 2},
    {"name": "Jupiter", "type": "Gas Giant", "diameter_km": 139820, "distance_from_sun_au": 5.20, "orbital_period_days": 4332.6, "moons": 95},
    {"name": "Saturn", "type": "Gas Giant", "diameter_km": 116460, "distance_from_sun_au": 9.58, "orbital_period_days": 10759.2, "moons": 146},
    {"name": "Uranus", "type": "Ice Giant", "diameter_km": 50724, "distance_from_sun_au": 19.20, "orbital_period_days": 30685.4, "moons": 28},
    {"name": "Neptune", "type": "Ice Giant", "diameter_km": 49244, "distance_from_sun_au": 30.05, "orbital_period_days": 60189.0, "moons": 16},
    {"name": "Pluto", "type": "Dwarf Planet", "diameter_km": 2377, "distance_from_sun_au": 39.48, "orbital_period_days": 90560.0, "moons": 5}
]

@router.get("")
def get_planets():
    return {
        "count": len(PLANET_CATALOGUE),
        "bodies": PLANET_CATALOGUE
    }
