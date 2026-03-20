#!/usr/bin/env python
"""
Isotropic Ordinary Kriging with AIC-based variogram model selection.
Uses Akaike Information Criterion to select the best variogram model.

Author: Clay Caldwell, Syracuse University & US Army ERDC-CHL

20 MAR 2026
"""

import numpy as np
# from pathlib import Path
# import verde as vd
# import geopandas as gpd
import matplotlib.pyplot as plt
import gstools as gs
from pykrige.ok import OrdinaryKriging
from shapely import contains_xy
import rasterio
from rasterio.transform import from_origin
import json
# from rasterio.mask import mask
# from shapely.geometry import box
import gc

class OK_AIC:
    """Generate kriged surfaces with AIC-based variogram model selection."""
    
    __slots__ = [
        'boundary', 'grid_res', 'vario_models', 'plot_outputs', 'coordinates', 'Z', 'max_lag', 'fit_model', 'fit_aic',  'ranking', 
        'residuals', 'krige_model', 'x_grid', 'y_grid', 'krige_pred', 'krige_var', 'transform', 'prediction_tif_path', 
        'uncertainty_tif_path', 'meta', '_crs', '_fold_spacing', '_empirical_bins', '_empirical_gamma', '_vario_estimator'
    ]
    
    def __init__(
        self, 
        vario_models, 
        x, 
        y, 
        z, 
        boundary, 
        pred_out_path, 
        var_out_path, 
        vario_estimator = 'cressie', 
        grid_res=10.0, 
        plot_outputs=False
    ):
        
        self.boundary = boundary
        self._crs = self.boundary.crs  # Cache CRS for later
        self.grid_res = grid_res
        self.vario_models = vario_models
        self.plot_outputs = plot_outputs
        self._vario_estimator = vario_estimator
        self.prediction_tif_path = pred_out_path
        self.uncertainty_tif_path = var_out_path

        # Store as float32 to reduce memory by 50%
        self.coordinates = (np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
        self.Z = np.asarray(z, dtype=np.float32)

        # Pre-compute range values
        x_range = float(self.coordinates[0].max() - self.coordinates[0].min())
        y_range = float(self.coordinates[1].max() - self.coordinates[1].min())

        # Maximum lag for variogram modeling is set to half the range of entire dataset. You may increase or decrease this by manulating the denominator
        self.max_lag = np.sqrt(x_range**2 + y_range**2) / 2

        self.fit_model = None
        self.fit_aic = None  # AIC score instead of pseudo-R² for variogram selection
        self.ranking = None
        self.residuals = None
        self.krige_model = None
        self.x_grid = None
        self.y_grid = None
        self.krige_pred = None
        self.krige_var = None
        self.transform = None
        self._empirical_bins = None
        self._empirical_gamma = None

    def _calculate_aic(self, model, bin_center, gamma):
        """
        Calculate AIC for a fitted variogram model.
        
        Parameters
        ----------
        model : gstools.CovModel
            Fitted variogram model
        bin_center : array
            Bin centers from empirical variogram
        gamma : array
            Empirical variogram values
        
        Returns
        -------
        float
            AIC score (lower is better)
        """
        # Get predicted values from the fitted model
        gamma_pred = model.variogram(bin_center)
        
        # Calculate residuals
        residuals = gamma - gamma_pred
        
        # Calculate RSS (Residual Sum of Squares)
        rss = np.sum(residuals**2)
        
        # Number of data points
        n = len(bin_center)
        
        # Number of parameters in the model
        # Most gstools models have 3 parameters: var (variance), len_scale (range), nugget
        # Some models may have additional parameters
        k = 3  # Default: variance, length scale, nugget
        
        # Handle models with additional parameters
        model_name = model.__class__.__name__.lower()
        if 'stable' in model_name:
            k = 4  # Includes shape parameter (alpha)
        elif 'matern' in model_name:
            k = 4  # Includes smoothness parameter (nu)
        
        # Calculate AIC using the formula: AIC = n * ln(RSS/n) + 2*k
        # Add small epsilon to avoid log(0)
        if rss < 1e-10:
            rss = 1e-10
        
        aic = n * np.log(rss / n) + 2 * k
        
        # Apply small sample correction (AICc) only when denominator is valid.
        if (n / k < 40) and (n > k + 1):
            aic = aic + (2 * k * (k + 1)) / (n - k - 1)
        
        return aic

    def _fit_variograms(self, max_evals=500000):
        """Fit variogram models and select best based on AIC."""
        bins = gs.variogram.standard_bins(
            pos=self.coordinates, dim=2, latlon=False,
            mesh_type="unstructured", max_dist=self.max_lag
        )
        
        data = self.Z
        bin_center, gamma = gs.vario_estimate(
            self.coordinates, data, bins, estimator=self._vario_estimator, latlon=False     # cressie or matheron
        )
        
        # Store for plotting
        self._empirical_bins = bin_center
        self._empirical_gamma = gamma
        
        vario_scores = {}
        best_aic, best_model = np.inf, None
        
        for name, model_class in self.vario_models.items():
            model = model_class(dim=2)
            
            try:
                # Fit the model (we don't need R² anymore)
                model.fit_variogram(bin_center, gamma, max_eval=max_evals)
                
                # Calculate AIC for this model
                aic = self._calculate_aic(model, bin_center, gamma)
                vario_scores[name] = aic
                
                # Lower AIC is better
                if aic < best_aic:
                    best_aic, best_model = aic, model
                    
            except Exception as e:
                print(f"Warning: Failed to fit {name} model: {e}")
                vario_scores[name] = np.inf
        
        self.fit_model = best_model
        self.fit_aic = best_aic
        # Rank by AIC (ascending - lower is better)
        self.ranking = sorted(vario_scores.items(), key=lambda x: x[1])

    def _krige(self):
        data = self.Z

        self.krige_model = OrdinaryKriging(
            self.coordinates[0], self.coordinates[1], data,
            variogram_model=self.fit_model,
            coordinates_type = 'euclidean'
        )

        # Pre-compute bounds
        x_min, x_max = float(self.coordinates[0].min()), float(self.coordinates[0].max())
        y_min, y_max = float(self.coordinates[1].min()), float(self.coordinates[1].max())

        # Build grid with consistent bounds (float32)
        nx = int(np.ceil((x_max - x_min) / self.grid_res)) + 1
        ny = int(np.ceil((y_max - y_min) / self.grid_res)) + 1
        self.x_grid = np.linspace(x_min, x_min + (nx - 1) * self.grid_res, nx, dtype=np.float32)
        self.y_grid = np.linspace(y_min, y_min + (ny - 1) * self.grid_res, ny, dtype=np.float32)

        self.transform = from_origin(
            x_min - self.grid_res / 2,
            y_max + self.grid_res / 2,
            self.grid_res, self.grid_res
        )

        polygon = self.boundary.geometry.iloc[0]
        nx, ny = len(self.x_grid), len(self.y_grid)

        z_final = np.full((ny, nx), np.nan, dtype=np.float32)
        ss_final = np.full((ny, nx), np.nan, dtype=np.float32)

        xx_full, yy_full = np.meshgrid(self.x_grid, self.y_grid)
        full_mask = contains_xy(polygon, xx_full, yy_full)

        z_pred, ss_final = self.krige_model.execute(
            style="masked", xpoints=self.x_grid, ypoints=self.y_grid,
            mask=~full_mask, backend="vectorized"
        )

        z_final = z_pred

        # Keep valid zero values; only masked cells become NaN.
        if np.ma.isMaskedArray(z_final):
            z_final = z_final.filled(np.nan)
        if np.ma.isMaskedArray(ss_final):
            ss_final = ss_final.filled(np.nan)

        self.krige_pred = np.asarray(z_final, dtype=np.float32)
        self.krige_var = np.asarray(ss_final, dtype=np.float32)

    def _write_output(self, pred_out_path=None, var_out_path=None, nodata_value=-9999.0):
        """Write COG rasters with AIC metadata."""
        if pred_out_path is not None:
            self.prediction_tif_path = pred_out_path
        if var_out_path is not None:
            self.uncertainty_tif_path = var_out_path

        z_export = np.flipud(np.nan_to_num(self.krige_pred, nan=nodata_value)).astype(np.float32)
        ss_export = np.flipud(np.nan_to_num(self.krige_var, nan=nodata_value)).astype(np.float32)
        
        meta = {
            'driver': 'GTiff',
            'height': z_export.shape[0],
            'width': z_export.shape[1],
            'count': 1,
            'dtype': 'float32',
            'crs': self._crs,
            'transform': self.transform,
            'nodata': nodata_value,
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
            'compress': 'deflate',
            'predictor': 3,
            'zlevel': 6
        }

        with rasterio.open(self.prediction_tif_path, 'w', **meta) as dst:
            dst.write(z_export, 1)
            dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.average)

            tags = {}
            
            # Add variogram model with AIC score
            tags['fit_variogram'] = str(self.fit_model)
            tags['variogram_aic'] = str(self.fit_aic)
            tags['selection_method'] = 'AIC'
            
            # Add ranking of all models
            tags['model_ranking'] = json.dumps([
                {'model': name, 'aic': float(score)} 
                for name, score in self.ranking
            ])

            dst.update_tags(**tags)

        # Write variance
        with rasterio.open(self.uncertainty_tif_path, 'w', **meta) as dst:
            dst.write(ss_export, 1)
            dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.average)
        
        # Free export arrays
        del z_export, ss_export

    def generate(self, pred_out_path=None, var_out_path=None):
        """Run complete Kriging workflow with AIC-based model selection."""
        
        self._fit_variograms()
        self._krige()

        if self.plot_outputs == True:
            self.plot_predictions()
            self.plot_variance()
            self.plot_variogram()

        self._write_output(pred_out_path, var_out_path)
        
        # Aggressive cleanup after writing
        self.krige_pred = None
        self.krige_var = None
        self.x_grid = None
        self.y_grid = None
        self.krige_model = None
        self.residuals = None

        gc.collect()
        
        print("Exported Ordinary Kriging (AIC):")
        print(f"  Best model: {self.ranking[0][0]} (AIC={self.fit_aic:.2f})")
        print(f"  {self.prediction_tif_path}")
        print(f"  {self.uncertainty_tif_path}")
        return self.prediction_tif_path, self.uncertainty_tif_path

    def plot_predictions(self):
        """Quick visualization of predictions."""
        if self.krige_pred is None:
            raise ValueError("Prediction data has been cleared. Call plot_predictions() before generate() completes.")
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        extent = [float(self.x_grid.min()), float(self.x_grid.max()), 
                  float(self.y_grid.min()), float(self.y_grid.max())]
        im = ax.imshow(self.krige_pred, origin="lower", extent=extent, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Depth')
        ax.set(xlabel='Easting', ylabel='Northing', 
               title=f'Kriging Interpolation (AIC Selection)\nModel: {self.ranking[0][0]} (AIC={self.fit_aic:.2f})')
        ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.show()

    def plot_variance(self):
        """Quick visualization of variance."""
        if self.krige_var is None:
            raise ValueError("Variance data has been cleared. Call plot_variance() before generate() completes.")
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        extent = [float(self.x_grid.min()), float(self.x_grid.max()), 
                  float(self.y_grid.min()), float(self.y_grid.max())]
        im = ax.imshow(self.krige_var, origin="lower", extent=extent, cmap='magma')
        plt.colorbar(im, ax=ax, label='Kriging Variance')
        ax.set(xlabel='Easting', ylabel='Northing',
               title=f'Kriging Uncertainty (AIC Selection)\nModel: {self.ranking[0][0]} (AIC={self.fit_aic:.2f})')
        ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.show()

    def plot_variogram(self):
        """Plot variogram cloud and best-fit variogram model in a 1x2 layout."""
        if self.fit_model is None:
            raise ValueError("No variogram model has been fitted yet. Call generate() first.")

        if not hasattr(self, '_empirical_bins') or not hasattr(self, '_empirical_gamma'):
            raise ValueError("Empirical variogram data not available. Call generate() first.")

        fig, (ax_cloud, ax_fit) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

        # Build a variogram cloud from random point pairs to avoid O(n^2) memory growth.
        x = self.coordinates[0]
        y = self.coordinates[1]
        z = self.Z
        n_pts = len(z)

        if n_pts > 1:
            max_pairs = min(50000, n_pts * (n_pts - 1) // 2)
            rng = np.random.default_rng(42)
            i = rng.integers(0, n_pts, size=max_pairs)
            j = rng.integers(0, n_pts, size=max_pairs)
            valid = i != j
            i = i[valid]
            j = j[valid]

            h = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            gamma_cloud = 0.5 * (z[i] - z[j]) ** 2

            ax_cloud.scatter(h, gamma_cloud, s=5, alpha=0.4, c='black', edgecolors='none')
            ax_cloud.set_title('Variogram Cloud', fontsize=12, fontweight='bold')
            ax_cloud.set_xlabel('Lag Distance', fontsize=11)
            ax_cloud.set_ylabel('Semivariance', fontsize=11)
            ax_cloud.grid(True, linestyle='--', alpha=0.3)
        else:
            ax_cloud.text(0.5, 0.5, 'Need at least 2 points for variogram cloud',
                          ha='center', va='center', transform=ax_cloud.transAxes)
            ax_cloud.set_title('Variogram Cloud', fontsize=12, fontweight='bold')

        # Plot empirical variogram and fitted model.
        ax_fit.scatter(self._empirical_bins, self._empirical_gamma,
                       c='blue', s=50, alpha=0.6, label='Empirical')

        x_model = np.linspace(0, self._empirical_bins.max(), 200)
        y_model = self.fit_model.variogram(x_model)

        ax_fit.plot(x_model, y_model, 'r-', linewidth=2,
                    label=f'Fitted: {self.ranking[0][0]}')

        sill = self.fit_model.var + self.fit_model.nugget
        ax_fit.axhline(y=sill, color='green', linestyle='--', linewidth=1.5,
                       alpha=0.7, label=f'Sill: {sill:.2f}')

        ax_fit.axhline(y=self.fit_model.nugget, color='orange', linestyle='--',
                       linewidth=1.5, alpha=0.7, label=f'Nugget: {self.fit_model.nugget:.2f}')

        ax_fit.axvline(x=self.fit_model.len_scale, color='purple', linestyle='--',
                       linewidth=1.5, alpha=0.7, label=f'Range: {self.fit_model.len_scale:.2f}')

        ax_fit.set_xlabel('Lag Distance', fontsize=11)
        ax_fit.set_ylabel('Semivariance', fontsize=11)
        ax_fit.set_title(f'Empirical + Fitted Variogram (AIC={self.fit_aic:.2f})', fontsize=12, fontweight='bold')
        ax_fit.legend(loc='lower right', fontsize=10)
        ax_fit.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()
