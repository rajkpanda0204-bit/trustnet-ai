document.addEventListener("DOMContentLoaded", () => {
    const reviewInput = document.getElementById("reviewText");
    const charCount = document.getElementById("charCount");
    const scanButton = document.getElementById("scanButton");
    const clearButton = document.getElementById("clearButton");

    const emptyState = document.getElementById("emptyState");
    const resultContent = document.getElementById("resultContent");
    const resultStatus = document.getElementById("resultStatus");
    const errorMessage = document.getElementById("errorMessage");

    // Check that the HTML elements exist.
    if (
        !reviewInput ||
        !charCount ||
        !scanButton ||
        !clearButton
    ) {
        console.error(
            "TRUSTNET AI: Required scanner elements were not found.",
            {
                reviewInput,
                charCount,
                scanButton,
                clearButton
            }
        );
        return;
    }

    // Update the character counter.
    function updateCharacterCount() {
        const length = reviewInput.value.length;

        charCount.textContent =
            `${length.toLocaleString()} / 10,000 characters`;

        // Prevent input beyond the backend limit.
        if (length > 10000) {
            charCount.style.color = "#ff6b6b";
        } else {
            charCount.style.color = "";
        }
    }

    reviewInput.addEventListener("input", updateCharacterCount);

    // Also initialise the counter when the page loads.
    updateCharacterCount();

    // Clear the review and reset the result panel.
    clearButton.addEventListener("click", (event) => {
        event.preventDefault();

        reviewInput.value = "";
        updateCharacterCount();

        if (emptyState) {
            emptyState.classList.remove("hidden");
        }

        if (resultContent) {
            resultContent.classList.add("hidden");
        }

        if (errorMessage) {
            errorMessage.classList.add("hidden");
            errorMessage.textContent = "";
        }

        if (resultStatus) {
            resultStatus.textContent = "AWAITING INPUT";
        }

        reviewInput.focus();
    });

    // Display an error in the result panel.
    function showError(message) {
        if (emptyState) {
            emptyState.classList.add("hidden");
        }

        if (resultContent) {
            resultContent.classList.add("hidden");
        }

        if (errorMessage) {
            errorMessage.textContent = message;
            errorMessage.classList.remove("hidden");
        }

        if (resultStatus) {
            resultStatus.textContent = "ANALYSIS FAILED";
        }
    }

    // Analyse the review using the Flask backend.
    scanButton.addEventListener("click", async (event) => {
        event.preventDefault();

        const review = reviewInput.value.trim();

        if (!review) {
            showError("Please enter a review before analysing.");
            reviewInput.focus();
            return;
        }

        if (review.length > 10000) {
            showError("The review cannot exceed 10,000 characters.");
            return;
        }

        scanButton.disabled = true;
        scanButton.textContent = "ANALYSING REVIEW...";

        if (errorMessage) {
            errorMessage.classList.add("hidden");
        }

        if (resultContent) {
            resultContent.classList.add("hidden");
        }

        if (emptyState) {
            emptyState.classList.remove("hidden");
        }

        if (resultStatus) {
            resultStatus.textContent = "PROCESSING";
        }

        try {
            const response = await fetch("/api/scan", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ review })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error || "The analysis request failed."
                );
            }

            document.getElementById("predictionLabel").textContent =
                data.prediction;

            document.getElementById("decisionScore").textContent =
                Number(data.decision_score).toFixed(4);

            document.getElementById("modelName").textContent =
                data.model;

            document.getElementById("reviewLength").textContent =
                `${data.review_length.toLocaleString()} characters`;

            document.getElementById("interpretation").textContent =
                data.interpretation;

            if (emptyState) {
                emptyState.classList.add("hidden");
            }

            if (resultContent) {
                resultContent.classList.remove("hidden");
            }

            if (resultStatus) {
                resultStatus.textContent = "ANALYSIS COMPLETE";
            }

        } catch (error) {
            console.error("TRUSTNET AI analysis error:", error);
            showError(error.message || "Something went wrong.");
        } finally {
            scanButton.disabled = false;
            scanButton.innerHTML =
                "<span>◈</span> ANALYSE REVIEW <span>↗</span>";
        }
    });

    // Load model comparison data.
    async function loadModelMetrics() {
        const chart = document.getElementById("modelChart");

        if (!chart) {
            return;
        }

        try {
            const response = await fetch("/api/model-metrics");

            if (!response.ok) {
                throw new Error("Could not load model metrics.");
            }

            const data = await response.json();
            const results = data.results || [];

            if (!results.length) {
                chart.textContent = "Model metrics are unavailable.";
                return;
            }

            if (typeof Plotly === "undefined") {
                chart.textContent = "Chart library could not be loaded.";
                return;
            }

            const names = results.map(item => item.model);
            const scores = results.map(
                item => Number(item.macro_f1) * 100
            );

            Plotly.newPlot(
                "modelChart",
                [{
                    x: names,
                    y: scores,
                    type: "bar",
                    marker: {
                        color: names.map(name =>
                            name === "Linear SVM"
                                ? "#56e5d2"
                                : "#a78bfa"
                        )
                    },
                    text: scores.map(
                        score => `${score.toFixed(2)}%`
                    ),
                    textposition: "outside",
                    hovertemplate:
                        "%{x}<br>Macro F1: %{y:.2f}%<extra></extra>"
                }],
                {
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    margin: {
                        t: 20,
                        r: 25,
                        b: 55,
                        l: 40
                    },
                    font: {
                        family: "DM Mono, monospace",
                        color: "#8797aa",
                        size: 10
                    },
                    yaxis: {
                        range: [0, 105],
                        ticksuffix: "%",
                        gridcolor: "#26313e",
                        zeroline: false
                    },
                    xaxis: {
                        gridcolor: "#26313e",
                        tickangle: -10
                    },
                    showlegend: false
                },
                {
                    responsive: true,
                    displayModeBar: false
                }
            );

        } catch (error) {
            console.error("Metrics loading error:", error);
            chart.textContent = "Model metrics are currently unavailable.";
        }
    }

    loadModelMetrics();

    // Check whether the backend and model are available.
    fetch("/api/health")
        .then(response => response.json())
        .then(data => {
            if (!data.model_loaded) {
                const accuracy = document.getElementById("accuracy");

                if (accuracy) {
                    accuracy.textContent = "OFFLINE";
                }
            }
        })
        .catch(error => {
            console.error("Health check failed:", error);
        });

    console.log("TRUSTNET AI: Scanner initialised successfully.");
});