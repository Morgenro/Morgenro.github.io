/**
 * ============================================
 * CYBERPUNK 2077 Additional JS Effects
 * Enhanced Interactions and Animations
 * ============================================
 */

(function() {
  'use strict';

  // ============================================
  // HEXAGON GRID BACKGROUND
  // ============================================
  function createHexagonGrid() {
    const container = document.createElement('div');
    container.id = 'hexagon-grid';
    container.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 0;
      opacity: 0.1;
    `;

    for (let i = 0; i < 20; i++) {
      const hex = document.createElement('div');
      hex.className = 'hexagon';
      hex.style.cssText = `
        position: absolute;
        top: ${Math.random() * 100}%;
        left: ${Math.random() * 100}%;
        width: ${20 + Math.random() * 40}px;
        height: ${(20 + Math.random() * 40) * 0.866}px;
        border: 1px solid ${['#FCE500', '#00F0FF', '#FF003C'][Math.floor(Math.random() * 3)]};
        clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
        animation: float ${5 + Math.random() * 5}s ease-in-out infinite;
        animation-delay: ${Math.random() * 3}s;
      `;
      container.appendChild(hex);
    }

    const style = document.createElement('style');
    style.textContent = `
      @keyframes float {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
      }
    `;
    document.head.appendChild(style);
    document.body.prepend(container);
  }

  // ============================================
  // DATA STREAMS (Vertical lines)
  // ============================================
  function createDataStreams() {
    const container = document.createElement('div');
    container.id = 'data-streams';
    container.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 1;
      opacity: 0.3;
    `;

    function createStream() {
      const stream = document.createElement('div');
      stream.style.cssText = `
        position: absolute;
        top: -100px;
        left: ${Math.random() * 100}%;
        width: 2px;
        height: ${50 + Math.random() * 100}px;
        background: linear-gradient(180deg, transparent, ${['#FCE500', '#00F0FF'][Math.floor(Math.random() * 2)]}, transparent);
        animation: stream-fall ${3 + Math.random() * 3}s linear;
      `;
      container.appendChild(stream);

      setTimeout(() => stream.remove(), 6000);
    }

    const style = document.createElement('style');
    style.textContent = `
      @keyframes stream-fall {
        0% { top: -100px; opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { top: 100%; opacity: 0; }
      }
    `;
    document.head.appendChild(style);
    document.body.appendChild(container);

    setInterval(createStream, 1000);
  }

  // ============================================
  // CORNER ORNAMENTS (2077 UI style) - DISABLED
  // ============================================
  // Removed to clean up the interface
  /* function addCornerOrnaments() {
    const corners = document.createElement('div');
    corners.id = 'corner-ornaments';
    corners.innerHTML = `
      <div class="corner top-left"></div>
      <div class="corner top-right"></div>
      <div class="corner bottom-left"></div>
      <div class="corner bottom-right"></div>
    `;

    const style = document.createElement('style');
    style.textContent = `
      #corner-ornaments .corner {
        position: fixed;
        width: 50px;
        height: 50px;
        border: 2px solid #FCE500;
        pointer-events: none;
        z-index: 9995;
        opacity: 0.6;
      }
      .corner.top-left {
        top: 20px;
        left: 20px;
        border-right: none;
        border-bottom: none;
      }
      .corner.top-right {
        top: 20px;
        right: 20px;
        border-left: none;
        border-bottom: none;
      }
      .corner.bottom-left {
        bottom: 20px;
        left: 20px;
        border-right: none;
        border-top: none;
      }
      .corner.bottom-right {
        bottom: 20px;
        right: 20px;
        border-left: none;
        border-top: none;
      }
    `;
    document.head.appendChild(style);
    document.body.appendChild(corners);
  } */

  // ============================================
  // STATUS BAR (Integrated with navigation)
  // ============================================
  function createStatusBar() {
    const statusBar = document.createElement('div');
    statusBar.id = 'cp-status-bar';
    statusBar.innerHTML = `
      <div class="status-item">
        <span class="status-label">[ SYSTEM ]</span>
        <span class="status-value" style="color:#00F0FF;">ONLINE</span>
      </div>
      <div class="status-item">
        <span class="status-label">[ NEURAL LINK ]</span>
        <span class="status-indicator"></span>
      </div>
      <div class="status-item">
        <span class="status-label">[ TIME ]</span>
        <span class="status-value" id="cp-time">--:--:--</span>
      </div>
    `;

    const style = document.createElement('style');
    style.textContent = `
      #cp-status-bar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 35px;
        background: rgba(10, 14, 39, 0.98);
        border-bottom: 1px solid #FCE500;
        display: flex;
        justify-content: space-around;
        align-items: center;
        font-family: 'Share Tech Mono', monospace;
        font-size: 10px;
        letter-spacing: 1.5px;
        color: #FCE500;
        z-index: 999;
        box-shadow: 0 2px 10px rgba(252, 229, 0, 0.2);
      }
      .status-item {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .status-label {
        color: #FCE500;
        font-weight: 800;
      }
      .status-value {
        color: #00F0FF;
        font-weight: 600;
      }
      /* Adjust nav bar to sit below status bar */
      #nav {
        margin-top: 35px !important;
      }
      /* Adjust page header to account for both bars */
      #page-header {
        padding-top: 45px !important;
      }
      @media (max-width: 768px) {
        #cp-status-bar {
          font-size: 7px;
          height: 28px;
        }
        #nav {
          margin-top: 28px !important;
        }
      }
    `;
    document.head.appendChild(style);
    document.body.prepend(statusBar);

    // Update time
    function updateTime() {
      const now = new Date();
      const time = now.toLocaleTimeString('en-US', { hour12: false });
      const timeEl = document.getElementById('cp-time');
      if (timeEl) timeEl.textContent = time;
    }
    updateTime();
    setInterval(updateTime, 1000);
  }

  // ============================================
  // GLITCH EFFECT ON RANDOM ELEMENTS
  // ============================================
  function initRandomGlitch() {
    setInterval(() => {
      const elements = document.querySelectorAll('h1, h2, h3, .article-title');
      if (elements.length === 0) return;

      const randomEl = elements[Math.floor(Math.random() * elements.length)];
      const originalText = randomEl.textContent;

      // Glitch characters
      const glitchChars = '!@#$%^&*()_+{}|:<>?~';
      let glitchedText = '';

      for (let i = 0; i < originalText.length; i++) {
        if (Math.random() > 0.9) {
          glitchedText += glitchChars[Math.floor(Math.random() * glitchChars.length)];
        } else {
          glitchedText += originalText[i];
        }
      }

      randomEl.textContent = glitchedText;

      setTimeout(() => {
        randomEl.textContent = originalText;
      }, 100);
    }, 8000);
  }

  // ============================================
  // CLICK RIPPLE EFFECT
  // ============================================
  function initClickRipple() {
    document.addEventListener('click', (e) => {
      const ripple = document.createElement('div');
      ripple.style.cssText = `
        position: fixed;
        width: 20px;
        height: 20px;
        border: 2px solid #FCE500;
        border-radius: 50%;
        pointer-events: none;
        z-index: 10000;
        animation: ripple-expand 1s ease-out forwards;
        left: ${e.clientX - 10}px;
        top: ${e.clientY - 10}px;
      `;

      const style = document.createElement('style');
      style.textContent = `
        @keyframes ripple-expand {
          0% {
            transform: scale(1);
            opacity: 1;
          }
          100% {
            transform: scale(20);
            opacity: 0;
          }
        }
      `;
      document.head.appendChild(style);
      document.body.appendChild(ripple);

      setTimeout(() => ripple.remove(), 1000);
    });
  }

  // ============================================
  // HOVER SOUND EFFECT (Visual feedback)
  // ============================================
  function initHoverEffects() {
    const interactiveElements = document.querySelectorAll('a, button, .card-widget, .recent-post-item');

    interactiveElements.forEach(el => {
      el.addEventListener('mouseenter', function() {
        // Add temporary glow
        this.style.transition = 'all 0.1s ease';
        this.style.filter = 'brightness(1.2) contrast(1.1)';
      });

      el.addEventListener('mouseleave', function() {
        this.style.filter = 'none';
      });
    });
  }

  // ============================================
  // SCROLL PROGRESS BAR
  // ============================================
  function createScrollProgress() {
    const progressBar = document.createElement('div');
    progressBar.id = 'scroll-progress';
    progressBar.style.cssText = `
      position: fixed;
      top: 30px;
      left: 0;
      height: 3px;
      background: linear-gradient(90deg, #FCE500, #00F0FF, #FF003C);
      width: 0%;
      z-index: 9998;
      transition: width 0.1s ease;
      box-shadow: 0 0 10px rgba(252, 229, 0, 0.8);
    `;
    document.body.appendChild(progressBar);

    window.addEventListener('scroll', () => {
      const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = (window.scrollY / windowHeight) * 100;
      progressBar.style.width = scrolled + '%';
    });
  }

  // ============================================
  // BREATHING EFFECT ON SIDEBAR
  // ============================================
  function initBreathingEffect() {
    const style = document.createElement('style');
    style.textContent = `
      @keyframes breathing {
        0%, 100% {
          box-shadow:
            0 0 20px rgba(255, 0, 60, 0.3),
            inset 0 0 40px rgba(255, 0, 60, 0.05);
        }
        50% {
          box-shadow:
            0 0 30px rgba(255, 0, 60, 0.5),
            inset 0 0 60px rgba(255, 0, 60, 0.1);
        }
      }
      #aside-content .card-widget {
        animation: breathing 3s ease-in-out infinite;
      }
    `;
    document.head.appendChild(style);
  }

  // ============================================
  // TYPING EFFECT FOR POST CONTENT
  // ============================================
  function initContentReveal() {
    const style = document.createElement('style');
    style.textContent = `
      .post-content p,
      .post-content li,
      .post-content h1,
      .post-content h2,
      .post-content h3 {
        opacity: 0;
        animation: fade-in-up 0.6s ease forwards;
      }
      @keyframes fade-in-up {
        from {
          opacity: 0;
          transform: translateY(20px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
    `;
    document.head.appendChild(style);

    // Stagger the animation
    const elements = document.querySelectorAll('.post-content p, .post-content li, .post-content h1, .post-content h2, .post-content h3');
    elements.forEach((el, index) => {
      el.style.animationDelay = `${index * 0.1}s`;
    });
  }

  // ============================================
  // INITIALIZE ALL ENHANCEMENTS
  // ============================================
  function init() {
    console.log('%c🔥 CYBERPUNK 2077 ENHANCEMENTS LOADED', 'color: #FF003C; font-size: 16px; font-weight: bold;');

    setTimeout(() => {
      createHexagonGrid();
      createDataStreams();
      // addCornerOrnaments(); // Disabled - corners removed
      createStatusBar();
      initRandomGlitch();
      initClickRipple();
      initHoverEffects();
      createScrollProgress();
      initBreathingEffect();
      initContentReveal();
    }, 200);
  }

  // Start
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
