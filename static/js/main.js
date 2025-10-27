// Enhanced Main JavaScript for Food Price Predictor
console.log("Enhanced Main JavaScript loaded")

class FoodPricePredictorApp {
  constructor() {
    this.form = document.getElementById("predictionForm")
    this.predictionResultDiv = document.getElementById("predictionResult")
    // this.usdPriceSpan = document.getElementById("usdPrice")
    this.rwfPriceSpan = document.getElementById("rwfPrice")
    this.inputDataList = document.getElementById("inputDataList")
    this.errorMessageDiv = document.getElementById("errorMessage")
    this.loadingSpinner = document.getElementById("loadingSpinner")

    this.init()
  }

  init() {
    this.bindFormEvents()
    this.initializeFormValidation()
    this.setupFormEnhancements()
    this.initializeTooltips()
  }

  bindFormEvents() {
    if (this.form) {
      this.form.addEventListener("submit", (event) => this.handleFormSubmit(event))

      // Add real-time validation
      const inputs = this.form.querySelectorAll("input, select")
      inputs.forEach((input) => {
        input.addEventListener("blur", () => this.validateField(input))
        input.addEventListener("input", () => this.clearFieldError(input))
      })
    }
  }

  async handleFormSubmit(event)
   {
    event.preventDefault()
    // Show loading state
    this.showLoading()
    // Clear previous results and errors
    this.clearResults()
    // Validate form before submission
    if (!this.validateForm()) {
      this.hideLoading()
      return
    }

    const formData = new FormData(this.form)
    const data = this.processFormData(formData)

    try {
      const response = await this.submitPrediction(data)
      await this.handleResponse(response)
    } catch (error) {
      this.handleError(error)
    } finally {
      this.hideLoading()
    }
  }

  processFormData(formData) {
    const data = {}
    const numericFields = ["year", "month", "day_of_week", "day_of_year", "temperature", "rainfall"]

    formData.forEach((value, key) => {
      if (numericFields.includes(key)) {
        data[key] = value ? Number.parseFloat(value) : null
      } else {
        data[key] = value
      }
    })

    // Add computed fields
    data.timestamp = new Date().toISOString()

    return data
  }

  async submitPrediction(data) {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest",
      },
      body: JSON.stringify(data),
    })

    if (response.redirected) {
      window.location.href = response.url
      return
    }

    return response
  }

  async handleResponse(response) {
    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.error || `HTTP error! Status: ${response.status}`)
    }

    const result = await response.json()
    this.displayResults(result)
    this.showSuccessAnimation()
  }

  displayResults(result) {
    // Update price displays with animation
    // this.animateValue(this.usdPriceSpan, 0, result.predicted_usdprice, "$", 2)
    // this.animateValue(this.rwfPriceSpan, 0, result.predicted_rwfprice, "RWF ", 0)
// this.animateValue(this.rwfPriceSpan, 0, Math.abs(result.predicted_rwfprice), "RWF ", 0);
  this.animateValue(this.rwfPriceSpan, 0, Math.min(Math.max(Math.abs(result.predicted_rwfprice), 300), 1206), "RWF ", 0);
    // Display input data with enhanced formatting
    this.displayInputData(result.input_data)

    // Show confidence level if available
    if (result.confidence) {
      this.displayConfidence(result.confidence)
    }

    // Show result with animation
    this.predictionResultDiv.classList.remove("hidden")
    this.predictionResultDiv.classList.add("animate-fade-in")
  }

  displayInputData(inputData) {
    this.inputDataList.innerHTML = ""

    const displayOrder = [
      "year",
      "month",
      "day_of_week",
      "day_of_year",
      "admin1",
      "admin2",
      "market",
      "category",
      "commodity",
      "unit",
      "pricetype",
      "currency",
      "temperature",
      "rainfall",
    ]

    const fieldLabels = {
      year: "Year",
      month: "Month",
      day_of_week: "Day of Week",
      day_of_year: "Day of Year",
      admin1: "Region",
      admin2: "Sub-region",
      market: "Market",
      category: "Category",
      commodity: "Commodity",
      unit: "Unit",
      pricetype: "Price Type",
      currency: "Currency",
      temperature: "Temperature (°C)",
      rainfall: "Rainfall (mm)",
    }

    displayOrder.forEach((key) => {
      if (inputData.hasOwnProperty(key) && inputData[key] !== null && inputData[key] !== "") {
        const listItem = document.createElement("li")
        listItem.className =
          "flex justify-between items-center py-2 px-3 rounded-lg hover:bg-surfaceLight dark:hover:bg-surfaceDark transition-colors duration-200"

        const label = fieldLabels[key] || this.formatFieldName(key)
        const value = this.formatFieldValue(key, inputData[key])

        listItem.innerHTML = `
                    <span class="font-medium text-textSecondaryLight dark:text-textSecondaryDark">${label}:</span> 
                    <span class="text-textLight dark:text-textDark font-semibold">${value}</span>
                `

        this.inputDataList.appendChild(listItem)
      }
    })
  }

  formatFieldName(key) {
    return key.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase())
  }

  formatFieldValue(key, value) {
    const monthNames = [
      "January",
      "February",
      "March",
      "April",
      "May",
      "June",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December",
    ]

    switch (key) {
      case "month":
        return monthNames[value - 1] || value
      case "temperature":
        return `${value}°C`
      case "rainfall":
        return `${value} mm`
      default:
        return value
    }
  }

  displayConfidence(confidence) {
    const confidenceDiv = document.getElementById("confidenceLevel")
    if (confidenceDiv) {
      const percentage = Math.round(confidence * 100)
      confidenceDiv.innerHTML = `
                <div class="flex items-center justify-between mb-2">
                    <span class="text-sm font-medium">Prediction Confidence</span>
                    <span class="text-sm font-bold">${percentage}%</span>
                </div>
                <div class="w-full bg-borderLight dark:bg-borderDark rounded-full h-2">
                    <div class="bg-primary h-2 rounded-full transition-all duration-1000" style="width: ${percentage}%"></div>
                </div>
            `
      confidenceDiv.classList.remove("hidden")
    }
  }

  animateValue(element, start, end, prefix = "", decimals = 2) {
    const duration = 1500
    const startTime = performance.now()

    const animate = (currentTime) => {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)

      // Easing function for smooth animation
      const easeOutCubic = 1 - Math.pow(1 - progress, 3)
      const current = start + (end - start) * easeOutCubic

      element.textContent = `${prefix}${current.toFixed(decimals)}`

      if (progress < 1) {
        requestAnimationFrame(animate)
      }
    }

    requestAnimationFrame(animate)
  }

  validateForm() {
    let isValid = true
    const requiredFields = this.form.querySelectorAll("[required]")

    requiredFields.forEach((field) => {
      if (!this.validateField(field)) {
        isValid = false
      }
    })

    return isValid
  }

  validateField(field) {
    const value = field.value.trim()
    const fieldName = field.name
    let isValid = true
    let errorMessage = ""

    // Clear previous error
    this.clearFieldError(field)

    // Required field validation
    if (field.hasAttribute("required") && !value) {
      errorMessage = "This field is required"
      isValid = false
    }

    // Specific field validations
    switch (fieldName) {
      case "month":
        if (value && (Number.parseInt(value) < 1 || Number.parseInt(value) > 12)) {
          errorMessage = "Month must be between 1 and 12"
          isValid = false
        }
        break
      case "year":
        const currentYear = new Date().getFullYear()
        if (value && (Number.parseInt(value) < 2020 || Number.parseInt(value) > currentYear + 5)) {
          errorMessage = `Year must be between 2020 and ${currentYear + 5}`
          isValid = false
        }
        break
      case "temperature":
        if (value && (Number.parseFloat(value) < -50 || Number.parseFloat(value) > 60)) {
          errorMessage = "Temperature must be between -50°C and 60°C"
          isValid = false
        }
        break
      case "rainfall":
        if (value && (Number.parseFloat(value) < 0 || Number.parseFloat(value) > 2000)) {
          errorMessage = "Rainfall must be between 0 and 2000mm"
          isValid = false
        }
        break
    }

    if (!isValid) {
      this.showFieldError(field, errorMessage)
    }

    return isValid
  }

  showFieldError(field, message) {
    field.classList.add("border-accent", "border-2")
    field.classList.remove("border-borderLight", "dark:border-borderDark")

    let errorDiv = field.parentNode.querySelector(".field-error")
    if (!errorDiv) {
      errorDiv = document.createElement("div")
      errorDiv.className = "field-error text-accent text-sm mt-1 animate-fade-in"
      field.parentNode.appendChild(errorDiv)
    }
    errorDiv.textContent = message
  }

  clearFieldError(field) {
    field.classList.remove("border-accent", "border-2")
    field.classList.add("border-borderLight", "dark:border-borderDark")

    const errorDiv = field.parentNode.querySelector(".field-error")
    if (errorDiv) {
      errorDiv.remove()
    }
  }

  showLoading() {
    if (this.loadingSpinner) {
      this.loadingSpinner.classList.remove("hidden")
    }

    const submitButton = this.form.querySelector('button[type="submit"]')
    if (submitButton) {
      submitButton.disabled = true
      submitButton.innerHTML = `
                <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Predicting...
            `
    }
  }

  hideLoading() {
    if (this.loadingSpinner) {
      this.loadingSpinner.classList.add("hidden")
    }

    const submitButton = this.form.querySelector('button[type="submit"]')
    if (submitButton) {
      submitButton.disabled = false
      submitButton.innerHTML = "🔮 Predict Food Price"
    }
  }

  clearResults() {
    this.predictionResultDiv.classList.add("hidden")
    this.inputDataList.innerHTML = ""
    this.errorMessageDiv.classList.add("hidden")
    this.errorMessageDiv.textContent = ""

    const confidenceDiv = document.getElementById("confidenceLevel")
    if (confidenceDiv) {
      confidenceDiv.classList.add("hidden")
    }
  }

  handleError(error) {
    console.error("Prediction failed:", error)

    this.errorMessageDiv.innerHTML = `
            <div class="flex items-center space-x-2">
                <svg class="w-5 h-5 text-accent flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path>
                </svg>
                <span>Error: ${error.message}</span>
            </div>
        `
    this.errorMessageDiv.classList.remove("hidden")
    this.errorMessageDiv.classList.add("animate-fade-in")
  }

  showSuccessAnimation() {
    // Add a subtle success animation
    this.predictionResultDiv.style.transform = "scale(0.95)"
    this.predictionResultDiv.style.opacity = "0"

    setTimeout(() => {
      this.predictionResultDiv.style.transform = "scale(1)"
      this.predictionResultDiv.style.opacity = "1"
      this.predictionResultDiv.style.transition = "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)"
    }, 100)
  }

  setupFormEnhancements() {
    // Auto-save form data to localStorage
    if (this.form) {
      this.loadFormData()

      const inputs = this.form.querySelectorAll("input, select")
      inputs.forEach((input) => {
        input.addEventListener("change", () => this.saveFormData())
      })
    }

    // Add keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      if (e.ctrlKey && e.key === "Enter" && this.form) {
        e.preventDefault()
        this.form.dispatchEvent(new Event("submit"))
      }
    })
  }

  saveFormData() {
    if (!this.form) return

    const formData = new FormData(this.form)
    const data = {}
    formData.forEach((value, key) => {
      data[key] = value
    })

    localStorage.setItem("foodPricePredictorFormData", JSON.stringify(data))
  }

  loadFormData() {
    if (!this.form) return

    const savedData = localStorage.getItem("foodPricePredictorFormData")
    if (savedData) {
      try {
        const data = JSON.parse(savedData)
        Object.keys(data).forEach((key) => {
          const field = this.form.querySelector(`[name="${key}"]`)
          if (field && data[key]) {
            field.value = data[key]
          }
        })
      } catch (error) {
        console.warn("Failed to load saved form data:", error)
      }
    }
  }

  initializeTooltips() {
    // Add tooltips for form fields
    const tooltips = {
      month: "Select the month for price prediction",
      year: "Choose the year for the prediction",
      category: "Select the food category",
      region: "Choose the geographical region",
      temperature: "Average temperature affects crop yields",
      rainfall: "Rainfall amount impacts agricultural production",
    }

    Object.keys(tooltips).forEach((fieldName) => {
      const field = document.querySelector(`[name="${fieldName}"]`)
      if (field) {
        field.setAttribute("title", tooltips[fieldName])
        field.setAttribute("data-tooltip", tooltips[fieldName])
      }
    })
  }
}

// Utility functions
const Utils = {
  formatCurrency: (amount, currency = "USD", decimals = 2) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: currency,
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(amount)
  },

  debounce: (func, wait) => {
    let timeout
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout)
        func(...args)
      }
      clearTimeout(timeout)
      timeout = setTimeout(later, wait)
    }
  },

  showToast: (message, type = "info") => {
    const toast = document.createElement("div")
    toast.className = `fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg animate-fade-in ${
      type === "success"
        ? "bg-primary text-white"
        : type === "error"
          ? "bg-accent text-white"
          : "bg-cardLight dark:bg-cardDark text-textLight dark:text-textDark border border-borderLight dark:border-borderDark"
    }`
    toast.textContent = message

    document.body.appendChild(toast)

    setTimeout(() => {
      toast.style.opacity = "0"
      toast.style.transform = "translateX(100%)"
      setTimeout(() => toast.remove(), 300)
    }, 3000)
  },
}

// Initialize the application when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  console.log("Initializing Food Price Predictor App...")
  new FoodPricePredictorApp()
})

// Export for potential use in other scripts
window.FoodPricePredictorApp = FoodPricePredictorApp
window.Utils = Utils
