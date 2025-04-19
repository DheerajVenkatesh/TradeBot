// Login Logic
document.getElementById('login-button').addEventListener('click', function() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    // Replace with your actual authentication logic
    if (username === 'user' && password === 'pass') {
        document.getElementById('login-page').style.display = 'none';
        document.getElementById('trading-page').style.display = 'flex';
    } else {
        alert('Invalid credentials');
    }
});

// Sidebar Tab Switching
document.querySelectorAll('.sidebar-button').forEach(button => {
    button.addEventListener('click', function() {
        const tabName = this.dataset.tab;
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.style.display = 'none';
        });
        document.getElementById(tabName).style.display = 'block';

        // Populate Data (Replace with your actual data fetching)
        if (tabName === 'positions') {
            document.getElementById('positions').innerHTML = `
                <p>Position 1: 10 shares of XYZ</p>
                <p>Position 2: 5 shares of ABC</p>
            `;
        } else if (tabName === 'portfolio') {
            document.getElementById('portfolio').innerHTML = `
                <p>Total Value: $10,000</p>
                <p>Cash: $5,000</p>
            `;
        }
    });
});