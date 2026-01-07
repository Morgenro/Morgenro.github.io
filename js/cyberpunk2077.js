/**
 * ============================================
 * CYBERPUNK 2077 JavaScript Effects
 * Night City Digital Experience
 * ============================================
 */

(function() {
  'use strict';

  // ============================================
  // DIGITAL RAIN EFFECT (Matrix-style)
  // ============================================
  function createDigitalRain() {
    const canvas = document.createElement('canvas');
    canvas.id = 'digital-rain-canvas';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '1';
    canvas.style.opacity = '0.15';
    document.body.prepend(canvas);

    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const chars = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン01';
    const charArray = chars.split('');
    const fontSize = 14;
    const columns = canvas.width / fontSize;
    const drops = Array(Math.floor(columns)).fill(1);

    function draw() {
      ctx.fillStyle = 'rgba(5, 8, 18, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.font = fontSize + 'px monospace';

      for (let i = 0; i < drops.length; i++) {
        // Alternate colors between yellow and cyan
        ctx.fillStyle = i % 2 === 0 ? '#FCE500' : '#00F0FF';

        const char = charArray[Math.floor(Math.random() * charArray.length)];
        ctx.fillText(char, i * fontSize, drops[i] * fontSize);

        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }
        drops[i]++;
      }
    }

    setInterval(draw, 50);

    window.addEventListener('resize', () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    });
  }

  // ============================================
  // MOUSE TRAIL GLOW EFFECT
  // ============================================
  function createMouseTrail() {
    const canvas = document.createElement('canvas');
    canvas.id = 'mouse-trail-canvas';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '9997';
    document.body.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    let particles = [];
    let mouse = { x: 0, y: 0 };

    class Particle {
      constructor(x, y) {
        this.x = x;
        this.y = y;
        this.size = Math.random() * 3 + 1;
        this.speedX = Math.random() * 3 - 1.5;
        this.speedY = Math.random() * 3 - 1.5;
        this.color = Math.random() > 0.5 ? '#FCE500' : '#00F0FF';
        this.life = 100;
      }

      update() {
        this.x += this.speedX;
        this.y += this.speedY;
        this.life -= 2;
        if (this.size > 0.1) this.size -= 0.05;
      }

      draw() {
        ctx.fillStyle = this.color;
        ctx.globalAlpha = this.life / 100;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();

        // Glow effect
        ctx.shadowBlur = 15;
        ctx.shadowColor = this.color;
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    function handleParticles() {
      for (let i = 0; i < particles.length; i++) {
        particles[i].update();
        particles[i].draw();

        if (particles[i].life <= 0) {
          particles.splice(i, 1);
          i--;
        }
      }
    }

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1;
      handleParticles();
      requestAnimationFrame(animate);
    }

    window.addEventListener('mousemove', (e) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;

      for (let i = 0; i < 3; i++) {
        particles.push(new Particle(mouse.x, mouse.y));
      }
    });

    window.addEventListener('resize', () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    });

    animate();
  }

  // ============================================
  // GLITCH EFFECT ON TITLE
  // ============================================
  function initGlitchEffect() {
    const title = document.querySelector('#site-name, .site-name, #site-title');
    if (!title) return;

    setInterval(() => {
      if (Math.random() > 0.95) {
        title.style.transform = `translate(${Math.random() * 4 - 2}px, ${Math.random() * 4 - 2}px)`;
        setTimeout(() => {
          title.style.transform = 'translate(0, 0)';
        }, 50);
      }
    }, 100);
  }

  // ============================================
  // SCAN LINE EFFECT
  // ============================================
  function createScanLine() {
    const scanLine = document.createElement('div');
    scanLine.id = 'scan-line';
    scanLine.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 3px;
      background: linear-gradient(to bottom,
        transparent,
        rgba(252, 229, 0, 0.8),
        transparent
      );
      box-shadow: 0 0 20px rgba(252, 229, 0, 0.8);
      pointer-events: none;
      z-index: 9996;
      animation: scan 4s linear infinite;
    `;

    const style = document.createElement('style');
    style.textContent = `
      @keyframes scan {
        0% { top: 0%; }
        100% { top: 100%; }
      }
    `;
    document.head.appendChild(style);
    document.body.appendChild(scanLine);
  }

  // ============================================
  // FLOATING PARTICLES BACKGROUND
  // ============================================
  function createFloatingParticles() {
    const canvas = document.createElement('canvas');
    canvas.id = 'particles-canvas';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '0';
    canvas.style.opacity = '0.3';
    document.body.prepend(canvas);

    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles = [];
    const particleCount = 50;

    class FloatingParticle {
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 2 + 0.5;
        this.speedX = Math.random() * 0.5 - 0.25;
        this.speedY = Math.random() * 0.5 - 0.25;
        this.color = ['#FCE500', '#00F0FF', '#FF003C'][Math.floor(Math.random() * 3)];
      }

      update() {
        this.x += this.speedX;
        this.y += this.speedY;

        if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
        if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;
      }

      draw() {
        ctx.fillStyle = this.color;
        ctx.shadowBlur = 10;
        ctx.shadowColor = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    for (let i = 0; i < particleCount; i++) {
      particles.push(new FloatingParticle());
    }

    function connectParticles() {
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 150) {
            ctx.strokeStyle = `rgba(0, 240, 255, ${1 - distance / 150})`;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }
    }

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach(particle => {
        particle.update();
        particle.draw();
      });
      connectParticles();
      requestAnimationFrame(animate);
    }

    window.addEventListener('resize', () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    });

    animate();
  }

  // ============================================
  // HOLOGRAM EFFECT ON IMAGES
  // ============================================
  function initHologramEffect() {
    const images = document.querySelectorAll('img');
    images.forEach(img => {
      img.addEventListener('mouseenter', function() {
        this.style.filter = 'hue-rotate(90deg) contrast(1.2) brightness(1.1)';
        this.style.transform = 'scale(1.05) translateZ(10px)';
      });

      img.addEventListener('mouseleave', function() {
        this.style.filter = 'none';
        this.style.transform = 'scale(1) translateZ(0)';
      });
    });
  }

  // ============================================
  // TYPING EFFECT FOR SITE SUBTITLE
  // ============================================
  function initTypingEffect() {
    const subtitle = document.querySelector('#site-subtitle, .site-subtitle');
    if (!subtitle) return;

    const text = subtitle.textContent;
    subtitle.textContent = '';
    let index = 0;

    function type() {
      if (index < text.length) {
        subtitle.textContent += text.charAt(index);
        index++;
        setTimeout(type, 100);
      }
    }

    setTimeout(type, 500);
  }

  // ============================================
  // CYBERPUNK CURSOR
  // ============================================
  function initCyberpunkCursor() {
    const cursor = document.createElement('div');
    cursor.id = 'cp-cursor';
    cursor.style.cssText = `
      position: fixed;
      width: 20px;
      height: 20px;
      border: 2px solid #FCE500;
      border-radius: 50%;
      pointer-events: none;
      z-index: 10001;
      mix-blend-mode: difference;
      transition: transform 0.1s ease;
      box-shadow: 0 0 20px rgba(252, 229, 0, 0.6);
    `;
    document.body.appendChild(cursor);

    const cursorDot = document.createElement('div');
    cursorDot.style.cssText = `
      position: fixed;
      width: 4px;
      height: 4px;
      background: #00F0FF;
      border-radius: 50%;
      pointer-events: none;
      z-index: 10002;
      box-shadow: 0 0 10px rgba(0, 240, 255, 0.8);
    `;
    document.body.appendChild(cursorDot);

    document.addEventListener('mousemove', (e) => {
      cursor.style.left = e.clientX - 10 + 'px';
      cursor.style.top = e.clientY - 10 + 'px';
      cursorDot.style.left = e.clientX - 2 + 'px';
      cursorDot.style.top = e.clientY - 2 + 'px';
    });

    document.addEventListener('mousedown', () => {
      cursor.style.transform = 'scale(0.8)';
    });

    document.addEventListener('mouseup', () => {
      cursor.style.transform = 'scale(1)';
    });
  }

  // ============================================
  // INIT AUDIO (V's Voice Line)
  // ============================================
  function addBootSound() {
    // Add a boot-up sound indicator
    const bootText = document.createElement('div');
    bootText.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      color: #FCE500;
      font-family: 'Share Tech Mono', monospace;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 2px;
      z-index: 10000;
      animation: fadeOut 3s forwards;
      text-shadow: 0 0 10px rgba(252, 229, 0, 0.8);
    `;
    bootText.textContent = '[ NEURAL LINK ESTABLISHED ]';

    const style = document.createElement('style');
    style.textContent = `
      @keyframes fadeOut {
        0%, 70% { opacity: 1; }
        100% { opacity: 0; }
      }
    `;
    document.head.appendChild(style);
    document.body.appendChild(bootText);

    setTimeout(() => bootText.remove(), 3000);
  }

  // ============================================
  // LOADING SCREEN OVERRIDE
  // ============================================
  function initLoadingScreen() {
    const loadingDiv = document.querySelector('#loading-box');
    if (loadingDiv) {
      loadingDiv.innerHTML = `
        <div style="color: #FCE500; font-family: 'Share Tech Mono', monospace; font-size: 20px; font-weight: 800; letter-spacing: 3px;">
          <div style="margin-bottom: 20px;">LOADING NIGHT CITY</div>
          <div style="font-size: 14px; color: #00F0FF;">[ INITIALIZING NEURAL INTERFACE ]</div>
        </div>
      `;
    }
  }

  // ============================================
  // RANDOM GLITCH ON ELEMENTS
  // ============================================
  function randomGlitchElements() {
    setInterval(() => {
      const elements = document.querySelectorAll('.recent-post-item, .card-widget');
      if (elements.length === 0) return;

      const randomElement = elements[Math.floor(Math.random() * elements.length)];
      randomElement.style.filter = 'hue-rotate(180deg) contrast(1.5)';

      setTimeout(() => {
        randomElement.style.filter = 'none';
      }, 100);
    }, 5000);
  }

  // ============================================
  // INITIALIZE ALL EFFECTS
  // ============================================
  function init() {
    console.log('%c🌃 CYBERPUNK 2077 THEME LOADED', 'color: #FCE500; font-size: 20px; font-weight: bold;');
    console.log('%cWelcome to Night City, Choom!', 'color: #00F0FF; font-size: 14px;');

    // Wait for DOM to be fully loaded
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        setTimeout(() => {
          createDigitalRain();
          createMouseTrail();
          createScanLine();
          createFloatingParticles();
          initGlitchEffect();
          initHologramEffect();
          initTypingEffect();
          initCyberpunkCursor();
          addBootSound();
          initLoadingScreen();
          randomGlitchElements();
        }, 100);
      });
    } else {
      setTimeout(() => {
        createDigitalRain();
        createMouseTrail();
        createScanLine();
        createFloatingParticles();
        initGlitchEffect();
        initHologramEffect();
        initTypingEffect();
        initCyberpunkCursor();
        addBootSound();
        initLoadingScreen();
        randomGlitchElements();
      }, 100);
    }
  }

  // Start initialization
  init();

})();
