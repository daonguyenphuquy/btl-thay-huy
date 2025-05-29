// Main JavaScript for MovieLens Recommender

document.addEventListener('DOMContentLoaded', function() {
    // Animate elements on scroll
    const animateOnScroll = function() {
        const elements = document.querySelectorAll('.movie-card, .feature-card, .stat-card, .chart-wrapper');

        elements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top;
            const windowHeight = window.innerHeight;

            if (elementPosition < windowHeight - 50) {
                element.style.opacity = '1';
                element.style.transform = 'translateY(0)';
            }
        });
    };

    // Set initial styles for animation
    const elementsToAnimate = document.querySelectorAll('.movie-card, .feature-card, .stat-card, .chart-wrapper');
    elementsToAnimate.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        element.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    });

    // Run animation on load and scroll
    setTimeout(animateOnScroll, 100);
    window.addEventListener('scroll', animateOnScroll);

    // Movie poster hover effect
    const moviePosters = document.querySelectorAll('.movie-poster');
    moviePosters.forEach(poster => {
        poster.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.05)';
            this.style.transition = 'transform 0.3s ease';
        });

        poster.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
        });
    });
});