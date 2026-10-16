
import { Component, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

// ============================================================
// STEP 1: Define the structure of each input feature
// ============================================================

interface FeatureField {
  name: string;
  label: string;
  type: 'number' | 'select' | 'binary';
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
}

// ============================================================
// STEP 2: Define the Flask prediction response
// ============================================================

interface PredictionResponse {
  model: string;
  probability: number;
  lr_probability: number;
  xgb_probability: number;
  shap_magnitude: number;
  disclaimer: string;
}

// ============================================================
// STEP 3: Angular component configuration
// ============================================================

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {

  title = 'CRC Research Prediction Dashboard';

  // Flask backend endpoint
  private readonly apiUrl = 'http://127.0.0.1:5000/predict';

  // Angular HTTP client
  private readonly http = inject(HttpClient);

  // Prediction status
  isLoading = false;
  predictionResult: PredictionResponse | null = null;
  predictionError = '';

  // ==========================================================
  // STEP 4: Define all 24 CRC input features
  // ==========================================================

  fields: FeatureField[] = [

    // --------------------------------------------------------
    // A. Demographic features
    // --------------------------------------------------------

    {
      name: 'Age',
      label: 'Age',
      type: 'number',
      min: 30,
      max: 89,
      step: 1
    },

    {
      name: 'Gender',
      label: 'Gender',
      type: 'select',
      options: ['F', 'M']
    },

    // --------------------------------------------------------
    // B. Lifestyle and medical history
    // --------------------------------------------------------

    {
      name: 'Smoking_History',
      label: 'Smoking History',
      type: 'select',
      options: ['No', 'Yes']
    },

    {
      name: 'Alcohol_Consumption',
      label: 'Alcohol Consumption',
      type: 'select',
      options: ['No', 'Yes']
    },

    {
      name: 'Obesity_BMI',
      label: 'Obesity / BMI Category',
      type: 'select',
      options: ['Normal', 'Obese', 'Overweight']
    },

    {
      name: 'Diabetes',
      label: 'Diabetes',
      type: 'select',
      options: ['No', 'Yes']
    },

    {
      name: 'Inflammatory_Bowel_Disease',
      label: 'Inflammatory Bowel Disease',
      type: 'select',
      options: ['No', 'Yes']
    },

    {
      name: 'Family_History',
      label: 'Family History',
      type: 'select',
      options: ['No', 'Yes']
    },

    {
      name: 'Diet_Risk',
      label: 'Diet Risk',
      type: 'select',
      options: ['High', 'Low', 'Moderate']
    },

    {
      name: 'Physical_Activity',
      label: 'Physical Activity',
      type: 'select',
      options: ['High', 'Low', 'Moderate']
    },

    // --------------------------------------------------------
    // C. Clinical symptoms
    // Binary encoding: 0 = No, 1 = Yes
    // --------------------------------------------------------

    {
      name: 'Abdominal Pain',
      label: 'Abdominal Pain',
      type: 'binary'
    },

    {
      name: 'Bleeding PR',
      label: 'Bleeding PR',
      type: 'binary'
    },

    {
      name: 'Weight Loss',
      label: 'Weight Loss',
      type: 'binary'
    },

    {
      name: 'Bowel Change',
      label: 'Bowel Change',
      type: 'binary'
    },

    {
      name: 'Tenesmus',
      label: 'Tenesmus',
      type: 'binary'
    },

    {
      name: 'Anemia related symptoms',
      label: 'Anemia Related Symptoms',
      type: 'binary'
    },

    {
      name: 'Abdominal Mass Palpable',
      label: 'Abdominal Mass Palpable',
      type: 'binary'
    },

    {
      name: 'PR Exam Suspicious',
      label: 'PR Exam Suspicious',
      type: 'binary'
    },

    {
      name: 'Pallor',
      label: 'Pallor',
      type: 'binary'
    },

    // --------------------------------------------------------
    // D. Laboratory investigations
    // --------------------------------------------------------

    {
      // IMPORTANT:
      // The trailing space in "Hemoglobin " is intentional.
      // It matches the saved ML preprocessing schema.
      name: 'Hemoglobin ',
      label: 'Hemoglobin',
      type: 'number',
      min: 7.4,
      max: 16.9,
      step: 0.1
    },

    {
      name: 'FIT/FOBT',
      label: 'FIT / FOBT',
      type: 'binary'
    },

    {
      name: 'CEA_Level',
      label: 'CEA Level',
      type: 'number',
      min: 0.4,
      max: 480.0,
      step: 0.1
    },

    {
      name: 'Liver function Test',
      label: 'Liver Function Test',
      type: 'select',
      options: ['Abnormal', 'Normal']
    },

    {
      name: 'serum Albumin',
      label: 'Serum Albumin',
      type: 'number',
      min: 2.0,
      max: 5.1,
      step: 0.1
    }

  ];

  // ==========================================================
  // STEP 5: Store all form values
  // ==========================================================

  formData: Record<string, string | number | null> = {};

  constructor() {
    this.resetForm();
  }

  // ==========================================================
  // STEP 6: Reset all 24 fields and previous prediction
  // ==========================================================

  resetForm(): void {

    this.formData = {};

    for (const field of this.fields) {
      this.formData[field.name] = null;
    }

    this.predictionResult = null;
    this.predictionError = '';
  }

  // ==========================================================
  // STEP 7: Send input data to Flask prediction API
  // ==========================================================

  previewInput(): void {

    if (this.isLoading) {
      return;
    }

    this.isLoading = true;
    this.predictionResult = null;
    this.predictionError = '';

    // Create a snapshot of the 24 input values
    const payload = { ...this.formData };

    console.log('====================================');
    console.log('CRC RESEARCH PREDICTION REQUEST');
    console.log('====================================');

    console.log('Number of features:', Object.keys(payload).length);
    console.log('Sending payload to Flask:', payload);

    this.http.post<PredictionResponse>(
      this.apiUrl,
      payload
    ).subscribe({

      // ------------------------------------------------------
      // Successful prediction
      // ------------------------------------------------------

      next: (response: PredictionResponse) => {

        this.predictionResult = response;
        this.isLoading = false;

        console.log('====================================');
        console.log('CRC RESEARCH PREDICTION RESPONSE');
        console.log('====================================');

        console.log('Model:', response.model);
        console.log('Final positive-class score:', response.probability);
        console.log('Logistic Regression:', response.lr_probability);
        console.log('XGBoost:', response.xgb_probability);
        console.log('SHAP magnitude:', response.shap_magnitude);
        console.log('Disclaimer:', response.disclaimer);

        
      },

      // ------------------------------------------------------
      // Failed prediction
      // ------------------------------------------------------

      error: (error) => {

        this.isLoading = false;

        this.predictionError =
          error.error?.error ||
          'Unable to connect to the Flask prediction API.';

        console.error('====================================');
        console.error('CRC PREDICTION ERROR');
        console.error('====================================');

        console.error('HTTP status:', error.status);
        console.error('Error details:', error);

        alert(
          'Prediction Request Failed!\n\n' +
          this.predictionError +
          '\n\nCheck that Flask is running on port 5000.'
        );

      }

    });

  }

}
