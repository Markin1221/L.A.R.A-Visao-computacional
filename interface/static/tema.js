// aplica o tema salvo antes da pagina aparecer (evita piscar com o tema errado)
(function () {
  var TEMAS = ["hud", "apple", "cyber", "engenharia", "aero"];
  var tema = "hud";
  try {
    var salvo = localStorage.getItem("lara_tema");
    if (TEMAS.indexOf(salvo) !== -1) tema = salvo;
  } catch (e) {}
  document.documentElement.setAttribute("data-tema", tema);
})();
