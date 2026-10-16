
import {
  ApplicationConfig,
  provideBrowserGlobalErrorListeners
} from '@angular/core';

import { provideRouter } from '@angular/router';

import { provideHttpClient } from '@angular/common/http';

import { routes } from './app.routes';

// ============================================================
// Angular Application Configuration
// ============================================================

export const appConfig: ApplicationConfig = {

  providers: [

    // Global Angular error handling
    provideBrowserGlobalErrorListeners(),

    // Angular routing
    provideRouter(routes),

    // Enable HTTP communication with Flask API
    provideHttpClient()

  ]

};
