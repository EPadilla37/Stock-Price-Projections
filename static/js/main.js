document.addEventListener('DOMContentLoaded', function() {
    const stockSelect = document.getElementById('stockSelect');
    const stockSymbol = document.getElementById('stockSymbol');
    const searchStock = document.getElementById('searchStock');
    const addStock = document.getElementById('addStock');
    const stockInfo = document.getElementById('stockInfo');
    const chartContainer = document.getElementById('chartContainer');
    const loadingSpinner = document.getElementById('loadingSpinner'); 

    let chart;

    // Populate stock dropdown
    function populateStockDropdown() {
        fetch('/get_stocks')
            .then(response => response.json())
            .then(data => {
                stockSelect.innerHTML = '<option selected>Choose a stock...</option>';
                data.stocks.forEach(stock => {
                    const option = document.createElement('option');
                    option.value = stock.symbol;
                    option.textContent = `${stock.symbol} - ${stock.name}`;
                    stockSelect.appendChild(option);
                });
            })
            .catch(error => {
                console.error('Error populating stock dropdown:', error);
                stockInfo.innerHTML = '<p class="text-danger">Error loading stocks. Please try again.</p>';
            });
    }

    // Initialize the dashboard
    populateStockDropdown();

    // Event listener for stock selection
    stockSelect.addEventListener('change', function() {
        const symbol = this.value;
        if (symbol !== 'Choose a stock...') {
            fetchStockData(symbol);
        }
    });

    // Event listener for stock search
    searchStock.addEventListener('click', function() {
        const symbol = stockSymbol.value.trim().toUpperCase();
        if (symbol) {
            fetch('/search_stock', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `symbol=${symbol}`
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    stockInfo.innerHTML = `<p class="text-danger">${data.error}</p>`;
                    addStock.style.display = 'none';
                } else {
                    stockInfo.innerHTML = `
                        <p><strong>${data.name}</strong></p>
                        <p>Symbol: ${data.symbol}</p>
                        <p>Sector: ${data.sector}</p>
                        <p>Industry: ${data.industry}</p>
                    `;
                    addStock.style.display = 'block';
                }
            })
            .catch(error => {
                console.error('Error searching stock:', error);
                stockInfo.innerHTML = '<p class="text-danger">Error searching stock. Please try again.</p>';
            });
        }
    });

    // Event listener for adding a stock
    addStock.addEventListener('click', function() {
        const symbol = stockSymbol.value.trim().toUpperCase();
        fetchStockData(symbol);
    });

    // Fetch stock data and update chart
    function fetchStockData(symbol) {
        // Show the loading spinner
        loadingSpinner.style.display = 'block';
        stockInfo.innerHTML = '<p class="text-info">Forecasting: This can take up to several minutes. Do not refresh the page.</p>';
        
        fetch('/select_stock', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `symbol=${symbol}`
        })
        .then(response => response.json())
        .then(data => {
            // Hide the loading spinner
            loadingSpinner.style.display = 'none';

            if (data.error) {
                stockInfo.innerHTML += `<p class="text-danger">${data.error}</p>`;
            } else {
                if (data.message === 'Data already exists') {
                    updateChart(data.historical_data, data.historical_forecast, data.current_forecast);
                    stockInfo.innerHTML = `<p class="text-success">Displaying data for ${symbol}</p>`;
                } else {
                    stockInfo.innerHTML += `<p class="text-success">${data.message}</p>`;
                    populateStockDropdown();
                    // Fetch the data again to get the historical and forecast data
                    fetchStockData(symbol);
                }
            }
        })
        .catch(error => {
            // Hide the loading spinner in case of error
            loadingSpinner.style.display = 'none';
            console.error('Error fetching stock data:', error);
            stockInfo.innerHTML = '<p class="text-danger">Error fetching stock data. Please try again.</p>';
        });
    }

    // Update the chart with new data
    function updateChart(historicalData, historicalForecast, currentForecast) {
        if (!chartContainer) {
            console.error('Chart container not found');
            return;
        }
    
        if (chart) {
            chart.destroy();
        }
    
        const labels = [
            ...historicalData.map(d => d.date),
            ...historicalForecast.map(d => d.date),
            ...currentForecast.map(d => d.date)
        ];
    
        chart = new Chart(chartContainer, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Historical Price',
                        data: historicalData.map(d => ({ x: d.date, y: d.price })),
                        borderColor: 'blue',
                        fill: false
                    },
                    {
                        label: 'Historical Forecast',
                        data: historicalForecast.map(d => ({ x: d.date, y: d.forecast })),
                        borderColor: 'green',
                        borderDash: [5, 5],
                        fill: false
                    },
                    // {
                    //     label: 'Actual Price (for Historical Forecast)',
                    //     data: historicalForecast.map(d => ({ x: d.date, y: d.actual })),
                    //     borderColor: 'purple',
                    //     pointStyle: 'circle',
                    //     pointRadius: 4,
                    //     fill: false
                    // },
                    {
                        label: 'Current Forecast',
                        data: currentForecast.map(d => ({ x: d.date, y: d.forecast })),
                        borderColor: 'red',
                        borderDash: [5, 5],
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            displayFormats: {
                                day: 'YYYY-MM-DD'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Price'
                        }
                    }
                }
            }
        });
    }
});
