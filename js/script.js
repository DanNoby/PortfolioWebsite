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
      if (!this.getAttribute("data-modal")) {
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
      }
    });
  });

  // Skill Progress Animation
  animateProgress("python-progress", 40, "#fca61f");
  animateProgress("javascript-progress", 20, "#6f34fe");
  animateProgress("cpp-progress", 30, "#20c997");
  animateProgress("java-progress", 10, "#3f396d");

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

  // Modal functionality
  const modalLinks = document.querySelectorAll("[data-modal]");
  const modals = document.querySelectorAll(".modal");
  const closeButtons = document.querySelectorAll(".close-modal");

  // Open modal when clicking on read more links
  modalLinks.forEach((link) => {
    link.addEventListener("click", function (e) {
      e.preventDefault();
      const modalId = this.getAttribute("data-modal");
      const modal = document.getElementById(modalId);

      if (modal) {
        modal.style.display = "block";
        document.body.style.overflow = "hidden"; // Prevent scrolling when modal is open
      }
    });
  });

  // Close modal when clicking on close button
  closeButtons.forEach((button) => {
    button.addEventListener("click", function () {
      const modal = this.closest(".modal");
      modal.style.display = "none";
      document.body.style.overflow = "auto"; // Enable scrolling again
    });
  });

  // Close modal when clicking outside the modal content
  window.addEventListener("click", function (e) {
    modals.forEach((modal) => {
      if (e.target === modal) {
        modal.style.display = "none";
        document.body.style.overflow = "auto"; // Enable scrolling again
      }
    });
  });

  // Close modal when pressing ESC key
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
      modals.forEach((modal) => {
        if (modal.style.display === "block") {
          modal.style.display = "none";
          document.body.style.overflow = "auto"; // Enable scrolling again
        }
      });
    }
  });

  // Form validation and submission
  const contactForm = document.querySelector(".contact-form form");

  if (contactForm) {
    contactForm.addEventListener("submit", function (e) {
      e.preventDefault(); // Prevent the default form submission

      // Validate form inputs
      const nameInput = contactForm.querySelector('input[placeholder="Name"]');
      const emailInput = contactForm.querySelector(
        'input[placeholder="E-mail"]'
      );
      const messageInput = contactForm.querySelector("textarea");

      // Simple validation
      if (validateForm(nameInput, emailInput, messageInput)) {
        // If validation passes
        alert("Form submitted successfully! Thank you for your message.");
        contactForm.reset(); // Clear the form
      } else {
        // If validation fails
        alert("Please enter valid details in all required fields.");
      }
    });
  }
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

// Form validation function
function validateForm(nameInput, emailInput, messageInput) {
  console.log("Name:", nameInput.value);
  console.log("Email:", emailInput.value);
  console.log("Message:", messageInput.value);

  // Check if name is valid
  if (!nameInput.value || nameInput.value.trim() === "") {
    console.log("Name validation failed");
    nameInput.focus();
    return false;
  }

  // Check email
  const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailInput.value || !emailPattern.test(emailInput.value)) {
    console.log("Email validation failed");
    emailInput.focus();
    return false;
  }

  // Check message
  if (!messageInput.value || messageInput.value.trim() === "") {
    console.log("Message validation failed");
    messageInput.focus();
    return false;
  }

  console.log("All validations passed");
  return true;
}
