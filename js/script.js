// Wait for the DOM to fully load
document.addEventListener("DOMContentLoaded", function () {
  // Mobile Menu Toggle
  const menuToggle = document.querySelector(".menu-toggle");
  const navMenu = document.querySelector(".nav-menu");

  if (menuToggle) {
    menuToggle.addEventListener("click", function () {
      navMenu.classList.toggle("active");
    });
  }

  // Smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();

      const targetId = this.getAttribute("href");
      const targetElement = document.querySelector(targetId);

      if (targetElement) {
        window.scrollTo({
          top: targetElement.offsetTop - 80,
          behavior: "smooth",
        });

        // Close mobile menu if open
        if (navMenu.classList.contains("active")) {
          navMenu.classList.remove("active");
        }
      }
    });
  });

  // Skill Progress Animation
  animateProgress("python-progress", 90, "#fca61f");
  animateProgress("javascript-progress", 75, "#6f34fe");
  animateProgress("cpp-progress", 80, "#20c997");
  animateProgress("java-progress", 30, "#3f396d");

  // Portfolio filtering
  const filterBtns = document.querySelectorAll(".filter-btn");
  const portfolioItems = document.querySelectorAll(".portfolio-item");

  filterBtns.forEach((btn) => {
    btn.addEventListener("click", function () {
      // Remove active class from all buttons
      filterBtns.forEach((filterBtn) => {
        filterBtn.classList.remove("active");
      });

      // Add active class to clicked button
      this.classList.add("active");

      const filterValue = this.getAttribute("data-filter");

      portfolioItems.forEach((item) => {
        if (filterValue === "all" || item.classList.contains(filterValue)) {
          item.style.display = "block";
        } else {
          item.style.display = "none";
        }
      });
    });
  });

  // Back to top button
  const backToTopBtn = document.getElementById("back-to-top");

  window.addEventListener("scroll", function () {
    if (
      document.body.scrollTop > 20 ||
      document.documentElement.scrollTop > 20
    ) {
      backToTopBtn.style.display = "block";
    } else {
      backToTopBtn.style.display = "none";
    }
  });

  backToTopBtn.addEventListener("click", function () {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  });

  // Sticky navbar functionality
  const navbar = document.getElementById("navbar");
  const sticky = navbar.offsetTop;

  window.addEventListener("scroll", function () {
    if (window.pageYOffset > sticky) {
      navbar.classList.add("sticky");
    } else {
      navbar.classList.remove("sticky");
    }
  });
});

// Function to animate skill progress
function animateProgress(elementId, endValue, color) {
  const progressElement = document.getElementById(elementId);
  if (!progressElement) return;

  const valueDisplay = progressElement.querySelector(".progress-value");
  let startValue = 0;

  const progress = setInterval(() => {
    startValue++;
    valueDisplay.textContent = `${startValue}%`;

    // Update progress background
    progressElement.style.background = `conic-gradient(${color} ${
      startValue * 3.6
    }deg, #ededed 0deg)`;

    if (startValue >= endValue) {
      clearInterval(progress);
    }
  }, 30);
}
