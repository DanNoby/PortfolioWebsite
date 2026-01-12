// Wait for the DOM to fully load
document.addEventListener("DOMContentLoaded", function () {
  // Initialize AOS animation library
  AOS.init({
    duration: 1000,
    offset: 100,
    once: false,
    mirror: false,
    easing: "ease-in-out",
  });

  // Mobile Menu Toggle
  const menuToggle = document.querySelector(".menu-toggle");
  const navMenu = document.querySelector(".nav-menu");
  const body = document.body;

  if (menuToggle) {
    menuToggle.addEventListener("click", function () {
      navMenu.classList.toggle("active");
      menuToggle.classList.toggle("active");

      // Toggle body scroll when menu is open
      if (navMenu.classList.contains("active")) {
        body.style.overflow = "hidden";
      } else {
        body.style.overflow = "auto";
      }
    });
  }

  // Close mobile menu when clicking outside
  document.addEventListener("click", function (e) {
    if (
      navMenu &&
      navMenu.classList.contains("active") &&
      !navMenu.contains(e.target) &&
      !menuToggle.contains(e.target)
    ) {
      navMenu.classList.remove("active");
      menuToggle.classList.remove("active");
      body.style.overflow = "auto";
    }
  });

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
        if (navMenu && navMenu.classList.contains("active")) {
          navMenu.classList.remove("active");
          menuToggle.classList.remove("active");
          body.style.overflow = "auto";
        }
      }
    });
  });

  // Navbar scroll effect
  const navbar = document.getElementById("navbar");
  window.addEventListener("scroll", function () {
    if (window.scrollY > 50) {
      navbar.classList.add("scrolled");
    } else {
      navbar.classList.remove("scrolled");
    }
  });

  // Skill Progress Animation using SVG for smoother animation
  function animateProgress(elementId, endValue, color) {
    const progressElement = document.getElementById(elementId);
    if (!progressElement) return;

    const valueDisplay = progressElement.querySelector(".progress-value");
    const circle = progressElement.querySelector(".progress-ring-circle");
    const radius = circle.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;

    circle.style.strokeDasharray = circumference;
    circle.style.strokeDashoffset = circumference;

    const setProgress = (percent) => {
      const offset = circumference - (percent / 100) * circumference;
      circle.style.strokeDashoffset = offset;
      valueDisplay.textContent = `${Math.round(percent)}%`;
    };

    // Animation timeline
    let startValue = 0;
    const duration = 1500; // 1.5 seconds
    const startTime = performance.now();

    function animate(currentTime) {
      const elapsedTime = currentTime - startTime;
      const progress = Math.min(elapsedTime / duration, 1);
      const currentValue = progress * endValue;

      setProgress(currentValue);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    }

    requestAnimationFrame(animate);
  }

  // Initialize progress animations when elements are in viewport
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          switch (entry.target.id) {
            case "python-progress":
              animateProgress("python-progress", 90, "#fca61f");
              break;
            case "javascript-progress":
              animateProgress("javascript-progress", 75, "#6f34fe");
              break;
            case "cpp-progress":
              animateProgress("cpp-progress", 80, "#20c997");
              break;
            case "java-progress":
              animateProgress("java-progress", 30, "#3f396d");
              break;
          }
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.5 }
  );

  // Observe skill progress elements
  document.querySelectorAll(".circular-progress").forEach((progress) => {
    observer.observe(progress);
  });

  // Portfolio filtering with smooth transitions
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

      // Apply filter with fade effect
      portfolioItems.forEach((item) => {
        if (filterValue === "all" || item.classList.contains(filterValue)) {
          item.style.opacity = "0";
          setTimeout(() => {
            item.style.display = "block";
            setTimeout(() => {
              item.style.opacity = "1";
            }, 50);
          }, 300);
        } else {
          item.style.opacity = "0";
          setTimeout(() => {
            item.style.display = "none";
          }, 300);
        }
      });
    });
  });

  // Back to top button
  const backToTopBtn = document.getElementById("back-to-top");

  window.addEventListener("scroll", function () {
    if (window.pageYOffset > 300) {
      backToTopBtn.style.opacity = "1";
      backToTopBtn.style.visibility = "visible";
    } else {
      backToTopBtn.style.opacity = "0";
      backToTopBtn.style.visibility = "hidden";
    }
  });

  backToTopBtn.addEventListener("click", function () {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  });

  // Modal functionality
  const modalLinks = document.querySelectorAll("[data-toggle='modal']");
  const modals = document.querySelectorAll(".modal");
  const closeButtons = document.querySelectorAll(".close-modal");

  modalLinks.forEach((link) => {
    link.addEventListener("click", function (e) {
      e.preventDefault();
      const targetModal = document.getElementById(
        this.getAttribute("data-target")
      );

      if (targetModal) {
        targetModal.style.display = "block";
        body.style.overflow = "hidden";

        // Add fade-in class
        setTimeout(() => {
          targetModal.querySelector(".modal-content").classList.add("fade-in");
        }, 10);
      }
    });
  });

  closeButtons.forEach((button) => {
    button.addEventListener("click", function () {
      const modal = this.closest(".modal");
      modal.querySelector(".modal-content").classList.remove("fade-in");

      setTimeout(() => {
        modal.style.display = "none";
        body.style.overflow = "auto";
      }, 300);
    });
  });

  // Close modal when clicking outside of content
  modals.forEach((modal) => {
    modal.addEventListener("click", function (e) {
      if (e.target === this) {
        modal.querySelector(".modal-content").classList.remove("fade-in");

        setTimeout(() => {
          modal.style.display = "none";
          body.style.overflow = "auto";
        }, 300);
      }
    });
  });

  // Close modal on escape key
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
      modals.forEach((modal) => {
        if (modal.style.display === "block") {
          modal.querySelector(".modal-content").classList.remove("fade-in");

          setTimeout(() => {
            modal.style.display = "none";
            body.style.overflow = "auto";
          }, 300);
        }
      });
    }
  });

  // Animated typing effect for home section
  function typeEffect() {
    const textElement = document.getElementById("type-text");
    if (!textElement) return;

    const phrases = [
      "Game Developer",
      "ML Engineer",
      "Web Developer",
      "Digital Artist",
    ];

    let phraseIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    let typeSpeed = 100;

    function type() {
      const currentPhrase = phrases[phraseIndex];

      if (isDeleting) {
        // Removing characters
        textElement.textContent = currentPhrase.substring(0, charIndex - 1);
        charIndex--;
        typeSpeed = 50;
      } else {
        // Adding characters
        textElement.textContent = currentPhrase.substring(0, charIndex + 1);
        charIndex++;
        typeSpeed = 150;
      }

      // Handle end of typing or deleting
      if (!isDeleting && charIndex === currentPhrase.length) {
        // Pause at end of phrase
        isDeleting = true;
        typeSpeed = 1500; // Pause before deleting
      } else if (isDeleting && charIndex === 0) {
        isDeleting = false;
        phraseIndex = (phraseIndex + 1) % phrases.length; // Move to next phrase
        typeSpeed = 500; // Pause before typing
      }

      setTimeout(type, typeSpeed);
    }

    // Start the typing effect
    setTimeout(type, 1000);
  }

  // Initialize typing effect
  typeEffect();

  // Enhanced form validation and submission
  const contactForm = document.getElementById("contactForm");
  if (contactForm) {
    contactForm.addEventListener("submit", function (e) {
      e.preventDefault();

      // Get form fields
      const name = document.getElementById("name").value;
      const email = document.getElementById("email").value;
      const message = document.getElementById("message").value;

      // Simple validation
      if (name.trim() === "" || email.trim() === "" || message.trim() === "") {
        showNotification("Please fill in all required fields.", "error");
        return;
      }

      // Email validation
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(email)) {
        showNotification("Please enter a valid email address.", "error");
        return;
      }

      // Simulate form submission - replace with actual submission code
      const submitBtn = contactForm.querySelector(".btn-submit");
      const originalText = submitBtn.innerHTML;

      submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
      submitBtn.disabled = true;

      // Simulate API call
      setTimeout(() => {
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
        contactForm.reset();
        showNotification(
          "Message sent successfully! I'll get back to you soon.",
          "success"
        );
      }, 2000);
    });
  }

  // Notification system
  function showNotification(message, type = "success") {
    // Create notification element
    const notification = document.createElement("div");
    notification.className = `notification ${type}`;
    notification.innerHTML = `
      <div class="notification-content">
        <i class="fas ${
          type === "success" ? "fa-check-circle" : "fa-exclamation-circle"
        }"></i>
        <span>${message}</span>
      </div>
      <button class="notification-close"><i class="fas fa-times"></i></button>
    `;

    // Add to DOM
    document.body.appendChild(notification);

    // Show with animation
    setTimeout(() => {
      notification.classList.add("show");
    }, 10);

    // Auto hide after 5 seconds
    setTimeout(() => {
      hideNotification(notification);
    }, 5000);

    // Close button
    const closeBtn = notification.querySelector(".notification-close");
    closeBtn.addEventListener("click", function () {
      hideNotification(notification);
    });
  }

  function hideNotification(notification) {
    notification.classList.remove("show");
    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 300);
  }

  // Parallax effect for shapes in background
  window.addEventListener("scroll", function () {
    const scrollPosition = window.pageYOffset;

    // Home section shapes
    const homeShapes = document.querySelectorAll(".home-shapes .shape");
    homeShapes.forEach((shape, index) => {
      const speed = (index + 1) * 0.1;
      if (shape) {
        shape.style.transform = `translateY(${scrollPosition * speed}px)`;
      }
    });

    // Contact section shapes
    const contactShapes = document.querySelectorAll(".contact-shapes .shape");
    contactShapes.forEach((shape, index) => {
      const speed = (index + 1) * 0.05;
      if (shape) {
        shape.style.transform = `translateY(${-scrollPosition * speed}px)`;
      }
    });
  });

  // Interactive hover effects for portfolio items
  const portfolioCards = document.querySelectorAll(".portfolio-card");

  portfolioCards.forEach((card) => {
    card.addEventListener("mouseenter", function () {
      const overlay = this.querySelector(".portfolio-overlay");
      const img = this.querySelector("img");

      if (overlay) overlay.style.opacity = "1";
      if (img) img.style.transform = "scale(1.1)";
    });

    card.addEventListener("mouseleave", function () {
      const overlay = this.querySelector(".portfolio-overlay");
      const img = this.querySelector("img");

      if (overlay) overlay.style.opacity = "0";
      if (img) img.style.transform = "scale(1)";
    });
  });

  // Add active class to navigation based on scroll position
  function updateNavActiveState() {
    const sections = document.querySelectorAll("section");
    const navLinks = document.querySelectorAll(".nav-menu a");

    sections.forEach((section) => {
      const sectionTop = section.offsetTop - 100;
      const sectionHeight = section.offsetHeight;
      const sectionId = section.getAttribute("id");

      if (
        window.scrollY >= sectionTop &&
        window.scrollY < sectionTop + sectionHeight
      ) {
        navLinks.forEach((link) => {
          link.classList.remove("active");
          if (link.getAttribute("href") === `#${sectionId}`) {
            link.classList.add("active");
          }
        });
      }
    });
  }

  window.addEventListener("scroll", updateNavActiveState);

  // Initial call to set active state on page load
  updateNavActiveState();
});

// Add CSS styles for notifications
const style = document.createElement("style");
style.textContent = `
  .notification {
    position: fixed;
    bottom: 30px;
    right: 30px;
    padding: 20px;
    background: white;
    box-shadow: 8px 8px 0 #000;
    border: 3px solid #000;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transform: translateY(100px);
    opacity: 0;
    transition: all 0.3s ease;
    z-index: 9999;
    max-width: 400px;
    font-family: 'Space Mono', monospace;
  }
  
  .notification.show {
    transform: translateY(0);
    opacity: 1;
  }
  
  .notification.success {
    border-left: 10px solid #00ff00;
  }
  
  .notification.error {
    border-left: 10px solid #ff00ff;
  }
  
  .notification-content {
    display: flex;
    align-items: center;
    gap: 15px;
  }
  
  .notification-content i {
    font-size: 1.4rem;
    color: #000 !important;
  }
  
  .notification-close {
    background: transparent;
    border: none;
    color: #000;
    cursor: pointer;
    margin-left: 15px;
    font-size: 1.2rem;
    font-weight: bold;
  }
  
  .notification-close:hover {
    color: #ff00ff;
  }
  
  @media (max-width: 576px) {
    .notification {
      bottom: 20px;
      right: 20px;
      left: 20px;
      max-width: none;
    }
  }
  
  /* Animation for menu toggle */
  .menu-toggle.active span:nth-child(1) {
    transform: translateY(10px) rotate(45deg);
  }
  
  .menu-toggle.active span:nth-child(2) {
    opacity: 0;
  }
  
  .menu-toggle.active span:nth-child(3) {
    transform: translateY(-10px) rotate(-45deg);
  }
  
  /* Animation for modal */
  .modal-content {
    transition: opacity 0.3s ease, transform 0.3s ease;
    opacity: 0;
    transform: translateY(-30px);
  }
  
  .modal-content.fade-in {
    opacity: 1;
    transform: translateY(0);
  }
`;

document.head.appendChild(style);
