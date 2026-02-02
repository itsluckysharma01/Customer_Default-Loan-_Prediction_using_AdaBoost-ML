document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("predictionForm");
  const resultsSection = document.getElementById("resultsSection");
  const resultsContent = document.getElementById("resultsContent");
  const submitButton = form.querySelector(".btn-submit");
  const buttonText = submitButton.querySelector(".btn-text");
  const loader = submitButton.querySelector(".loader");

  // Form submission handler
  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    // Disable submit button and show loader
    submitButton.disabled = true;
    buttonText.textContent = "Processing...";
    loader.style.display = "block";

    // Collect form data
    const formData = {
      customer_age: parseFloat(document.getElementById("customer_age").value),
      customer_income: parseFloat(
        document.getElementById("customer_income").value,
      ),
      home_ownership: document.getElementById("home_ownership").value,
      employment_duration: parseFloat(
        document.getElementById("employment_duration").value,
      ),
      loan_intent: document.getElementById("loan_intent").value,
      loan_grade: document.getElementById("loan_grade").value,
      loan_amnt: parseFloat(document.getElementById("loan_amnt").value),
      loan_int_rate: parseFloat(document.getElementById("loan_int_rate").value),
      loan_percent_income: parseFloat(
        document.getElementById("loan_percent_income").value,
      ),
      historical_default: document.getElementById("historical_default").value,
      credit_history_length: parseFloat(
        document.getElementById("credit_history_length").value,
      ),
    };

    try {
      // Send prediction request
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();

      if (result.success) {
        displayResults(result);
      } else {
        displayError(result.error || "An error occurred during prediction.");
      }
    } catch (error) {
      displayError(
        "Failed to connect to the server. Please ensure the server is running.",
      );
      console.error("Error:", error);
    } finally {
      // Re-enable submit button and hide loader
      submitButton.disabled = false;
      buttonText.textContent = "Predict Default Risk";
      loader.style.display = "none";
    }
  });

  // Display results
  function displayResults(result) {
    const isPredictedDefault = result.prediction === 1;
    const cardClass = isPredictedDefault ? "danger" : "success";
    const icon = isPredictedDefault ? "❌" : "✅";
    const riskLevel = result.risk_level;

    // Determine risk badge class
    let riskBadgeClass = "low";
    if (riskLevel.includes("Very High")) {
      riskBadgeClass = "very-high";
    } else if (riskLevel.includes("High")) {
      riskBadgeClass = "high";
    } else if (riskLevel.includes("Medium")) {
      riskBadgeClass = "medium";
    }

    resultsContent.innerHTML = `
            <div class="result-card ${cardClass}">
                <div class="result-header">
                    <div class="result-icon">${icon}</div>
                    <div class="result-title">
                        <h3>${result.prediction_label}</h3>
                        <p>Based on the provided customer and loan information</p>
                    </div>
                </div>
                
                <div class="risk-level">
                    <strong>Risk Assessment:</strong>
                    <span class="risk-badge ${riskBadgeClass}">${riskLevel}</span>
                </div>

                <div class="confidence-bars">
                    <h4 style="margin-bottom: 15px; color: var(--text-primary);">Confidence Levels</h4>
                    
                    <div class="confidence-item">
                        <div class="confidence-label">
                            <span>No Default Probability</span>
                            <span><strong>${result.confidence.no_default.toFixed(2)}%</strong></span>
                        </div>
                        <div class="confidence-bar-container">
                            <div class="confidence-bar no-default" style="width: ${result.confidence.no_default}%"></div>
                        </div>
                    </div>

                    <div class="confidence-item">
                        <div class="confidence-label">
                            <span>Default Probability</span>
                            <span><strong>${result.confidence.default.toFixed(2)}%</strong></span>
                        </div>
                        <div class="confidence-bar-container">
                            <div class="confidence-bar default" style="width: ${result.confidence.default}%"></div>
                        </div>
                    </div>
                </div>

                <div style="margin-top: 25px; padding-top: 20px; border-top: 1px solid var(--border-color);">
                    <h4 style="margin-bottom: 10px; color: var(--text-primary);">Recommendation</h4>
                    <p style="color: var(--text-secondary); line-height: 1.6;">
                        ${getRecommendation(isPredictedDefault, riskLevel, result.confidence.default)}
                    </p>
                </div>
            </div>
        `;

    // Show results section with smooth scroll
    resultsSection.style.display = "block";
    resultsSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // Display error message
  function displayError(errorMessage) {
    resultsContent.innerHTML = `
            <div class="error-message">
                ${errorMessage}
            </div>
        `;
    resultsSection.style.display = "block";
    resultsSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // Generate recommendation based on prediction
  function getRecommendation(isDefault, riskLevel, defaultProb) {
    if (isDefault) {
      if (defaultProb > 80) {
        return "Strong indication of default risk. Loan application should be rejected or require substantial collateral and guarantees. Consider requesting additional documentation and co-signers.";
      } else if (defaultProb > 60) {
        return "High risk of default detected. If proceeding with the loan, implement strict monitoring, require additional collateral, and consider a higher interest rate to offset risk.";
      } else {
        return "Moderate risk of default. Consider approving with enhanced terms: higher down payment, shorter loan tenure, or additional security measures.";
      }
    } else {
      if (defaultProb < 20) {
        return "Excellent credit profile with very low default risk. Customer qualifies for favorable loan terms, lower interest rates, and higher loan amounts.";
      } else if (defaultProb < 40) {
        return "Good credit standing with acceptable risk levels. Approve loan with standard terms and regular monitoring protocols.";
      } else {
        return "Acceptable risk profile. Approve with standard terms but maintain regular account monitoring and periodic reviews.";
      }
    }
  }

  // Auto-calculate loan to income ratio
  const incomeInput = document.getElementById("customer_income");
  const loanAmountInput = document.getElementById("loan_amnt");
  const loanPercentInput = document.getElementById("loan_percent_income");

  function calculateLoanPercent() {
    const income = parseFloat(incomeInput.value) || 0;
    const loanAmount = parseFloat(loanAmountInput.value) || 0;

    if (income > 0 && loanAmount > 0) {
      const ratio = loanAmount / income;
      loanPercentInput.value = ratio.toFixed(4);
    }
  }

  incomeInput.addEventListener("input", calculateLoanPercent);
  loanAmountInput.addEventListener("input", calculateLoanPercent);

  // Form reset handler
  form.addEventListener("reset", function () {
    resultsSection.style.display = "none";
    resultsContent.innerHTML = "";
  });

  // Input validation and formatting
  const numberInputs = form.querySelectorAll('input[type="number"]');
  numberInputs.forEach((input) => {
    input.addEventListener("input", function () {
      if (this.value < 0) {
        this.value = 0;
      }
    });
  });
});
