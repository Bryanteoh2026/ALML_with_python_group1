const form = document.getElementById("churnForm");

if (form) {
    const fields = Array.from(form.querySelectorAll("input[required], select[required]"));
    const progressFill = document.getElementById("progressFill");
    const completedCount = document.getElementById("completedCount");
    const progressText = document.getElementById("progressText");
    const sampleButton = document.getElementById("sampleButton");
    const resetButton = document.getElementById("resetButton");

    const tips = [
        "Good start! Add the customer’s service experience next.",
        "Halfway there — the customer signal is becoming clearer.",
        "Almost ready! Complete the final few clues.",
        "All clues found. The prediction button is ready!"
    ];

    const sampleCustomer = {
        MonthlyRevenue: "82.50",
        MonthlyMinutes: "540",
        TotalRecurringCharge: "65",
        OverageMinutes: "62",
        RoamingCalls: "3",
        DroppedCalls: "8",
        CustomerCareCalls: "5",
        MonthsInService: "14",
        CurrentEquipmentDays: "610",
        HandsetWebCapable: "Yes",
        HandsetRefurbished: "No",
        RetentionCalls: "2",
        RetentionOffersAccepted: "0",
        IncomeGroup: "6",
        HandsetPrice: "120",
        MadeCallToRetentionTeam: "Yes",
        CreditRating: "3-Good",
        PrizmCode: "Suburban",
        MaritalStatus: "Unknown"
    };

    function hasValue(field) {
        return field.value.trim() !== "";
    }

    function updateFieldStyle(field) {
        const wrapper = field.closest(".field");
        const complete = hasValue(field);

        wrapper.classList.toggle("completed", complete);
        wrapper.classList.remove("invalid");
    }

    function updateSnapshot() {
        const revenue = document.getElementById("MonthlyRevenue").value;
        const minutes = document.getElementById("MonthlyMinutes").value;
        const months = document.getElementById("MonthsInService").value;
        const dropped = Number(document.getElementById("DroppedCalls").value || 0);
        const careCalls = Number(document.getElementById("CustomerCareCalls").value || 0);

        document.getElementById("revenuePreview").textContent = revenue
            ? `$${Number(revenue).toFixed(2)} / month`
            : "Not entered";

        document.getElementById("minutesPreview").textContent = minutes
            ? `${Number(minutes).toLocaleString()} minutes`
            : "Not entered";

        document.getElementById("servicePreview").textContent = months
            ? `${Number(months)} months with telco`
            : "Not entered";

        let mood = "Waiting for clues";
        if (dropped > 5 || careCalls > 3) {
            mood = "Needs attention ⚠️";
        } else if (dropped > 0 || careCalls > 0) {
            mood = "A few service issues";
        } else if (hasValue(document.getElementById("DroppedCalls"))) {
            mood = "Service looks calm ✨";
        }
        document.getElementById("moodPreview").textContent = mood;
    }

    function updateProgress() {
        const completed = fields.filter(hasValue).length;
        const percentage = (completed / fields.length) * 100;

        completedCount.textContent = completed;
        progressFill.style.width = `${percentage}%`;

        if (completed === 0) {
            progressText.textContent = "Let’s begin! Pick your first customer clue.";
        } else if (percentage < 40) {
            progressText.textContent = tips[0];
        } else if (percentage < 75) {
            progressText.textContent = tips[1];
        } else if (completed < fields.length) {
            progressText.textContent = tips[2];
        } else {
            progressText.textContent = tips[3];
        }

        fields.forEach(updateFieldStyle);
        updateSnapshot();
    }

    fields.forEach((field) => {
        field.addEventListener("input", updateProgress);
        field.addEventListener("change", updateProgress);
    });

    form.addEventListener("submit", (event) => {
        const emptyFields = fields.filter((field) => !hasValue(field));

        if (emptyFields.length > 0) {
            event.preventDefault();
            emptyFields.forEach((field) => field.closest(".field").classList.add("invalid"));
            progressText.textContent = `Complete ${emptyFields.length} more field(s) to unlock the prediction.`;
            emptyFields[0].focus();
        }
    });

    sampleButton.addEventListener("click", () => {
        Object.entries(sampleCustomer).forEach(([fieldName, value]) => {
            const field = document.getElementById(fieldName);
            if (field) {
                field.value = value;
            }
        });
        updateProgress();
        progressText.textContent = "Sample customer loaded! Change any clue or predict now.";
    });

    resetButton.addEventListener("click", () => {
        window.setTimeout(() => {
            fields.forEach((field) => {
                field.closest(".field").classList.remove("completed", "invalid");
            });
            updateProgress();
        }, 0);
    });

    updateProgress();
}

const analyseAgain = document.getElementById("analyseAgain");
if (analyseAgain && form) {
    analyseAgain.addEventListener("click", () => {
        form.reset();
        form.querySelectorAll(".field").forEach((field) => {
            field.classList.remove("completed", "invalid");
        });
        window.scrollTo({ top: form.offsetTop - 20, behavior: "smooth" });
        window.setTimeout(() => window.location.assign("/"), 350);
    });
}

function createConfetti() {
    const colours = ["#6757e5", "#ff8a65", "#ffd166", "#1f9d78", "#8fd3ff"];

    for (let index = 0; index < 55; index += 1) {
        const piece = document.createElement("span");
        piece.className = "confetti-piece";
        piece.style.left = `${Math.random() * 100}vw`;
        piece.style.background = colours[index % colours.length];
        piece.style.animationDelay = `${Math.random() * 0.6}s`;
        piece.style.animationDuration = `${1.8 + Math.random() * 1.3}s`;
        document.body.appendChild(piece);
        window.setTimeout(() => piece.remove(), 3500);
    }
}

const result = document.body.dataset.result;
const resultSection = document.getElementById("resultSection");

if (resultSection) {
    window.setTimeout(() => {
        resultSection.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 180);
}

if (result === "Likely to Stay") {
    createConfetti();
}
