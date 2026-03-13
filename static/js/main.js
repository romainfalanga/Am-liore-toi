document.addEventListener('DOMContentLoaded', function () {

  // --- Mobile menu toggle ---
  var toggle = document.getElementById('menu-toggle');
  var nav = document.getElementById('nav-links');
  if (toggle && nav) {
    toggle.addEventListener('click', function () {
      nav.classList.toggle('open');
      toggle.innerHTML = nav.classList.contains('open') ? '&#10005;' : '&#9776;';
    });
    // Close menu on link click
    nav.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', function () {
        nav.classList.remove('open');
        toggle.innerHTML = '&#9776;';
      });
    });
  }

  // --- Header scroll effect ---
  var header = document.getElementById('site-header');
  if (header) {
    var onScroll = function () {
      if (window.scrollY > 50) {
        header.classList.add('scrolled');
      } else {
        header.classList.remove('scrolled');
      }
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  // --- Scroll reveal animations ---
  var reveals = document.querySelectorAll('.reveal');
  if (reveals.length > 0 && 'IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: '0px 0px -40px 0px'
    });

    reveals.forEach(function (el, i) {
      // Stagger delay for grid items
      el.style.transitionDelay = (i % 3) * 0.12 + 's';
      observer.observe(el);
    });
  } else {
    // Fallback: show all
    reveals.forEach(function (el) { el.classList.add('visible'); });
  }

  // --- Smooth page transition feel ---
  document.body.style.opacity = '0';
  document.body.style.transition = 'opacity 0.4s ease';
  requestAnimationFrame(function () {
    document.body.style.opacity = '1';
  });

});
