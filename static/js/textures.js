/* Orbit Overload Static Textures Cache */
(function() {
  var isFile = (typeof window !== 'undefined' && window.location.protocol === 'file:');
  var isPage = (typeof window !== 'undefined' && window.location.pathname.indexOf('/pages/') !== -1);
  var base = isFile ? (isPage ? '../static/textures/' : 'static/textures/') : '/static/textures/';
  
  window.TX = {
    "earth": base + "earth.jpg",
    "lights": base + "lights.jpg",
    "clouds": base + "clouds.jpg",
    "moon": base + "moon.jpg",
    "mercury": base + "mercury.jpg",
    "venus": base + "venus.jpg",
    "mars": base + "mars.jpg",
    "jupiter": base + "jupiter.jpg",
    "saturn": base + "saturn.jpg",
    "uranus": base + "uranus.jpg",
    "neptune": base + "neptune.jpg",
    "ringc": base + "ringc.jpg",
    "ringa": base + "ringa.jpg"
  };
})();
