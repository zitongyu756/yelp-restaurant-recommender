document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const minStarsInput = document.getElementById('min-stars');
    const starsValue = document.getElementById('stars-value');
    const priceFilter = document.getElementById('filter-price');
    const hoodFilter = document.getElementById('filter-neighborhood');
    const loadMoreBtn = document.getElementById('load-more-btn');
    const suggestionBtns = document.querySelectorAll('.suggestion-btn');
    
    const resultsHeader = document.getElementById('results-header');
    const resultsContainer = document.getElementById('results-container');
    const resultsMeta = document.getElementById('results-meta');
    const loader = document.getElementById('loader');
    const template = document.getElementById('result-card-template');

    // State for client-side filtering
    window.allResults = [];
    window.filteredResults = [];
    window.displayLimit = 10;
    window.currentQuery = "";
    window.fetchTime = 0;

    // Filter event listeners (instant client-side filter)
    minStarsInput.addEventListener('input', (e) => {
        starsValue.textContent = parseFloat(e.target.value).toFixed(1);
        applyFilters();
    });
    priceFilter.addEventListener('change', applyFilters);
    hoodFilter.addEventListener('change', applyFilters);

    loadMoreBtn.addEventListener('click', () => {
        window.displayLimit += 10;
        renderResults();
    });

    // Handle suggestion clicks
    suggestionBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            searchInput.value = btn.textContent;
            performSearch();
        });
    });

    // Handle Enter key in search
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch();
        }
    });

    // Handle search button click
    searchBtn.addEventListener('click', performSearch);

    async function performSearch() {
        let query = searchInput.value.trim();
        if (!query) return;

        const reqHood = hoodFilter.value;
        if (reqHood !== "any") {
            const hoodText = hoodFilter.options[hoodFilter.selectedIndex].text;
            query = query + " in " + hoodText;
        }

        window.currentQuery = query;

        // Show loading state
        resultsContainer.innerHTML = '';
        resultsHeader.style.display = 'none';
        loadMoreBtn.style.display = 'none';
        loader.style.display = 'flex';

        try {
            const start = performance.now();
            // Fetch 30 candidates to allow client-side filtering
            const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&top_k=30`);
            const data = await response.json();
            const end = performance.now();
            window.fetchTime = end - start;

            if (data.status === 'error') {
                showError(data.message);
                return;
            }

            // Reset pagination and store data
            window.allResults = data.results;
            window.displayLimit = 10;
            applyFilters();
            
        } catch (err) {
            showError("Failed to connect to the recommendation engine.");
            console.error(err);
        } finally {
            loader.style.display = 'none';
        }
    }

    function applyFilters() {
        const minStars = parseFloat(minStarsInput.value);
        const maxPrice = priceFilter.value;
        const reqHood = hoodFilter.value.toLowerCase().trim();

        window.filteredResults = window.allResults.filter(r => {
            // 1. Star check
            if (r.stars < minStars) return false;
            
            // 2. Price check (e.g. if maxPrice is $$, filter out $$$ and $$$$)
            if (maxPrice !== "any") {
                if (!r.price) return false; // N/A
                if (r.price.length > maxPrice.length) return false;
            }
            
            return true;
        });

        renderResults();
    }

    function renderResults() {
        resultsContainer.innerHTML = '';
        
        if (window.filteredResults.length === 0) {
            resultsContainer.innerHTML = '<div style="color: #9BA1A6; padding: 20px;">No restaurants found matching your criteria. Try loosening the filters!</div>';
            resultsHeader.style.display = 'flex';
            resultsMeta.textContent = `0 results for "${window.currentQuery}"`;
            loadMoreBtn.style.display = 'none';
            return;
        }

        resultsHeader.style.display = 'flex';
        resultsMeta.textContent = `${window.filteredResults.length} matching results · ${(window.fetchTime / 1000).toFixed(2)}s`;

        const toDisplay = window.filteredResults.slice(0, window.displayLimit);

        toDisplay.forEach((r, index) => {
            const clone = template.content.cloneNode(true);
            
            // Rank
            clone.querySelector('.rank-number').textContent = `#${index + 1}`;
            
            // Core info
            clone.querySelector('.restaurant-name').textContent = r.name;
            clone.querySelector('.score-value').textContent = `${r.match_pct}%`;
            
            clone.querySelector('.rating').textContent = `★ ${r.stars.toFixed(1)}`;
            clone.querySelector('.reviews').textContent = `${r.review_count} reviews`;
            clone.querySelector('.price').textContent = r.price || "N/A";
            
            // Location fallback logic
            let locText = "";
            if (r.address && r.address !== "nan") {
                locText = r.address;
            } else if (r.neighborhood && r.neighborhood !== "nan") {
                locText = r.neighborhood;
            } else if (r.city && r.city !== "nan") {
                locText = r.city;
            }
            clone.querySelector('.location-text').textContent = locText;
            
            // Categories
            clone.querySelector('.restaurant-categories').textContent = r.categories;
            
            // Badges
            const badgesContainer = clone.querySelector('.badges-container');
            const expLower = r.explanation.toLowerCase();
            if (expLower.includes("quiet")) {
                badgesContainer.innerHTML += `<span class="badge">Quiet</span>`;
            }
            if (expLower.includes("romantic")) {
                badgesContainer.innerHTML += `<span class="badge">Romantic</span>`;
            }
            if (expLower.includes("budget")) {
                badgesContainer.innerHTML += `<span class="badge">Budget Friendly</span>`;
            }
            
            // Format explanation (parse markdown bold **text**)
            let formattedExp = r.explanation.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            // Parse newlines
            formattedExp = formattedExp.replace(/\n/g, '<br>');
            clone.querySelector('.explanation-text').innerHTML = formattedExp;

            resultsContainer.appendChild(clone);
        });

        // Toggle load more button
        if (window.filteredResults.length > window.displayLimit) {
            loadMoreBtn.style.display = 'inline-block';
        } else {
            loadMoreBtn.style.display = 'none';
        }
    }

    function showError(msg) {
        resultsContainer.innerHTML = `<div style="color: #F43F5E; padding: 20px; background: rgba(244,63,94,0.1); border-radius: 8px;">Error: ${msg}</div>`;
        resultsHeader.style.display = 'none';
    }
});
