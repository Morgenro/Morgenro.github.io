// 生活部分交互功能
document.addEventListener('DOMContentLoaded', function() {
  // 初始化小说评分
  const ratingStars = document.querySelectorAll('.novel-rating .star');
  ratingStars.forEach(star => {
    star.addEventListener('click', function() {
      const rating = parseInt(this.getAttribute('data-rating')) || 0;
      const container = this.closest('.novel-rating');
      
      container.querySelectorAll('.star').forEach((s, i) => {
        s.classList.toggle('filled', i < rating);
      });
    });
  });

  // 音乐播放器初始化
  if (document.getElementById('music-player')) {
    const ap = new APlayer({
      container: document.getElementById('music-player'),
      audio: [{
        name: '示例音乐',
        artist: '艺术家',
        url: '/music/sample.mp3',
        cover: '/img/cover.jpg'
      }]
    });
  }
});