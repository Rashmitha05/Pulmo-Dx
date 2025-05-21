const diagnosisBtns = document.querySelectorAll('.diagnosis-options button');
const uploadSections = document.querySelectorAll('.upload-section');

// Add event listeners to diagnosis buttons
diagnosisBtns.forEach(btn => {
  btn.addEventListener('click', function() {
    diagnosisBtns.forEach(otherBtn => otherBtn.classList.remove('active'));
    this.classList.add('active');

    // Hide all upload sections
    uploadSections.forEach(section => section.classList.remove('active'));

    // Show the upload section for the selected diagnosis
    const selectedDiagnosis = this.id.replace('-btn', '-upload');
    document.getElementById(selectedDiagnosis).classList.add('active');

    // (Optional) Enable submit button only when a file is selected
    const submitBtn = document.querySelector(`#${selectedDiagnosis} button[type="submit"]`);
    const fileInput = document.querySelector(`#${selectedDiagnosis} input[type="file"]`);
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        submitBtn.disabled = false;
      } else {
        submitBtn.disabled = true;
      }
    });
  });
});
