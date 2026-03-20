#!/usr/bin/env python
"""
Regression Kriging with configurable scikit-learn regression model.

Uses PyKrige RegressionKriging to combine a deterministic regression model
with kriged residuals for spatial interpolation.

Author: Clay Caldwell, Syracuse University & US Army ERDC-CHL

20 MAR 2026
"""

import gc
import json
from pathlib import Path

import gstools as gs
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pykrige.ok import OrdinaryKriging
from pykrige.rk import RegressionKriging
from rasterio.transform import from_origin
from shapely import contains_xy
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


class RK_AIC:
    """Generate regression-kriging surfaces with selectable sklearn regressors."""

    __slots__ = [
        'boundary', 'grid_res', 'coordinates', 'Z', '_crs', 'prediction_tif_path',
        'uncertainty_tif_path', 'regression_model_input', 'regression_model_params',
        'variogram_model', 'vario_models', '_vario_estimator', 'max_lag', 'fit_model',
        'fit_aic', 'ranking', '_empirical_bins', '_empirical_gamma', 'n_closest_points',
        'plot_outputs', 'rk_model', 'reg_model', 'x_grid', 'y_grid', 'transform',
        'rk_pred', 'rk_var'
    ]

    def __init__(
        self,
        x,
        y,
        z,
        boundary,
        pred_out_path,
        var_out_path,
        regression_model='linear',
        regression_model_params=None,
        vario_models=None,
        vario_estimator='cressie',
        variogram_model='spherical',
        n_closest_points=12,
        grid_res=10.0,
        plot_outputs=False,
    ):
        self.boundary = boundary
        self._crs = boundary.crs
        self.grid_res = grid_res
        self.prediction_tif_path = pred_out_path
        self.uncertainty_tif_path = var_out_path
        self.regression_model_input = regression_model
        self.regression_model_params = regression_model_params or {}
        self.variogram_model = variogram_model
        self.vario_models = self._resolve_vario_models(vario_models, variogram_model)
        self._vario_estimator = vario_estimator
        self.n_closest_points = n_closest_points
        self.plot_outputs = plot_outputs

        self.coordinates = (np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
        self.Z = np.asarray(z, dtype=np.float32)

        x_range = float(self.coordinates[0].max() - self.coordinates[0].min())
        y_range = float(self.coordinates[1].max() - self.coordinates[1].min())
        self.max_lag = np.sqrt(x_range**2 + y_range**2) / 2

        self.fit_model = None
        self.fit_aic = None
        self.ranking = None
        self._empirical_bins = None
        self._empirical_gamma = None

        self.rk_model = None
        self.reg_model = None
        self.x_grid = None
        self.y_grid = None
        self.transform = None
        self.rk_pred = None
        self.rk_var = None

    def _resolve_vario_models(self, vario_models, variogram_model):
        if vario_models is not None:
            return vario_models

        available = {
            'gaussian': gs.Gaussian,
            'exponential': gs.Exponential,
            'matern': gs.Matern,
            'stable': gs.Stable,
            'rational': gs.Rational,
            'circular': gs.Circular,
            'spherical': gs.Spherical,
            'superspherical': gs.SuperSpherical,
            'jbessel': gs.JBessel,
        }

        key = str(variogram_model).strip().lower()
        if key in available:
            return {key.capitalize(): available[key]}

        # Fallback to broad AIC search when model name is unrecognized.
        return {
            'Gaussian': gs.Gaussian,
            'Exponential': gs.Exponential,
            'Matern': gs.Matern,
            'Stable': gs.Stable,
            'Rational': gs.Rational,
            'Circular': gs.Circular,
            'Spherical': gs.Spherical,
            'SuperSpherical': gs.SuperSpherical,
            'JBessel': gs.JBessel,
        }

    def _calculate_aic(self, model, bin_center, gamma):
        gamma_pred = model.variogram(bin_center)
        residuals = gamma - gamma_pred
        rss = np.sum(residuals**2)
        n = len(bin_center)

        k = 3
        model_name = model.__class__.__name__.lower()
        if 'stable' in model_name or 'matern' in model_name:
            k = 4

        if rss < 1e-10:
            rss = 1e-10

        aic = n * np.log(rss / n) + 2 * k
        if (n / k < 40) and (n > k + 1):
            aic = aic + (2 * k * (k + 1)) / (n - k - 1)

        return aic

    def _fit_variograms(self, residuals, max_evals=500000):
        bins = gs.variogram.standard_bins(
            pos=self.coordinates, dim=2, latlon=False,
            mesh_type='unstructured', max_dist=self.max_lag
        )

        bin_center, gamma = gs.vario_estimate(
            self.coordinates, residuals, bins, estimator=self._vario_estimator, latlon=False
        )

        self._empirical_bins = bin_center
        self._empirical_gamma = gamma

        vario_scores = {}
        best_aic, best_model = np.inf, None

        for name, model_class in self.vario_models.items():
            model = model_class(dim=2)
            try:
                model.fit_variogram(bin_center, gamma, max_eval=max_evals)
                aic = self._calculate_aic(model, bin_center, gamma)
                vario_scores[name] = aic
                if aic < best_aic:
                    best_aic, best_model = aic, model
            except Exception as e:
                print(f"Warning: Failed to fit {name} model: {e}")
                vario_scores[name] = np.inf

        if best_model is None:
            raise ValueError('No variogram model fit succeeded for RK residuals.')

        self.fit_model = best_model
        self.fit_aic = best_aic
        self.ranking = sorted(vario_scores.items(), key=lambda x: x[1])

    def _resolve_regression_model(self):
        """Resolve sklearn regressor from a name or a user-provided estimator."""
        model = self.regression_model_input
        params = self.regression_model_params

        if hasattr(model, 'fit') and hasattr(model, 'predict'):
            return clone(model)

        if not isinstance(model, str):
            raise ValueError(
                "regression_model must be either a sklearn estimator or one of: "
                "'linear', 'ridge', 'lasso', 'elasticnet', 'random_forest', "
                "'gradient_boosting', 'svr', 'knn'."
            )

        key = model.strip().lower()
        model_map = {
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'elasticnet': ElasticNet,
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'svr': SVR,
            'knn': KNeighborsRegressor,
        }

        if key not in model_map:
            raise ValueError(
                "Unsupported regression_model. Choose one of: "
                "'linear', 'ridge', 'lasso', 'elasticnet', 'random_forest', "
                "'gradient_boosting', 'svr', 'knn'."
            )

        return model_map[key](**params)

    def _fit(self):
        self.reg_model = self._resolve_regression_model()

        # Use x,y as predictors by default for trend estimation.
        predictors = np.column_stack((self.coordinates[0], self.coordinates[1])).astype(np.float64)
        coords = predictors.copy()

        self.reg_model.fit(predictors, self.Z.astype(np.float64))
        reg_pred = self.reg_model.predict(predictors)
        residuals = self.Z.astype(np.float64) - reg_pred

        self._fit_variograms(residuals)

        self.rk_model = RegressionKriging(
            regression_model=self.reg_model,
            variogram_model=self.fit_model,
            n_closest_points=self.n_closest_points,
        )
        self.rk_model.fit(predictors, coords, self.Z.astype(np.float64))

    def _predict_grid(self):
        x_min, x_max = float(self.coordinates[0].min()), float(self.coordinates[0].max())
        y_min, y_max = float(self.coordinates[1].min()), float(self.coordinates[1].max())

        nx = int(np.ceil((x_max - x_min) / self.grid_res)) + 1
        ny = int(np.ceil((y_max - y_min) / self.grid_res)) + 1
        self.x_grid = np.linspace(x_min, x_min + (nx - 1) * self.grid_res, nx, dtype=np.float32)
        self.y_grid = np.linspace(y_min, y_min + (ny - 1) * self.grid_res, ny, dtype=np.float32)

        self.transform = from_origin(
            x_min - self.grid_res / 2,
            y_max + self.grid_res / 2,
            self.grid_res,
            self.grid_res,
        )

        polygon = self.boundary.geometry.iloc[0]
        xx_full, yy_full = np.meshgrid(self.x_grid, self.y_grid)
        full_mask = contains_xy(polygon, xx_full, yy_full)

        points_in = np.column_stack((xx_full[full_mask], yy_full[full_mask])).astype(np.float64)
        z_pred_flat = self.rk_model.predict(points_in, points_in)

        pred_grid = np.full(xx_full.shape, np.nan, dtype=np.float32)
        pred_grid[full_mask] = z_pred_flat.astype(np.float32)
        self.rk_pred = pred_grid

        # Estimate residual kriging variance for uncertainty output.
        train_points = np.column_stack((self.coordinates[0], self.coordinates[1])).astype(np.float64)
        reg_train = self.rk_model.regression_model.predict(train_points)
        residuals = self.Z.astype(np.float64) - reg_train

        ok_resid = OrdinaryKriging(
            self.coordinates[0].astype(np.float64),
            self.coordinates[1].astype(np.float64),
            residuals,
            variogram_model=self.fit_model,
        )
        _, ss = ok_resid.execute(
            style='masked',
            xpoints=self.x_grid,
            ypoints=self.y_grid,
            mask=~full_mask,
            backend='vectorized',
        )
        if np.ma.isMaskedArray(ss):
            ss = ss.filled(np.nan)
        self.rk_var = np.asarray(ss, dtype=np.float32)

    def _write_output(self, pred_out_path=None, var_out_path=None, nodata_value=-9999.0):
        if pred_out_path is not None:
            self.prediction_tif_path = pred_out_path
        if var_out_path is not None:
            self.uncertainty_tif_path = var_out_path

        Path(self.prediction_tif_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.uncertainty_tif_path).parent.mkdir(parents=True, exist_ok=True)

        z_export = np.flipud(np.nan_to_num(self.rk_pred, nan=nodata_value)).astype(np.float32)
        ss_export = np.flipud(np.nan_to_num(self.rk_var, nan=nodata_value)).astype(np.float32)

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
            'zlevel': 6,
        }

        with rasterio.open(self.prediction_tif_path, 'w', **meta) as dst:
            dst.write(z_export, 1)
            dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.average)
            dst.update_tags(
                method='RegressionKriging',
                fit_variogram=str(self.fit_model),
                variogram_aic=str(self.fit_aic),
                regression_model=str(self.rk_model.regression_model),
                model_ranking=json.dumps([
                    {'model': name, 'aic': float(score)}
                    for name, score in self.ranking
                ]),
            )

        with rasterio.open(self.uncertainty_tif_path, 'w', **meta) as dst:
            dst.write(ss_export, 1)
            dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.average)

    def generate(self, pred_out_path=None, var_out_path=None):
        self._fit()
        self._predict_grid()

        if self.plot_outputs is True:
            self.plot_predictions()
            self.plot_variance()
            self.plot_variogram()

        self._write_output(pred_out_path, var_out_path)

        self.rk_pred = None
        self.rk_var = None
        self.x_grid = None
        self.y_grid = None

        gc.collect()

        print('Exported Regression Kriging:')
        print(f'  Regressor: {self.rk_model.regression_model}')
        print(f'  Best variogram: {self.ranking[0][0]} (AIC={self.fit_aic:.2f})')
        print(f'  {self.prediction_tif_path}')
        print(f'  {self.uncertainty_tif_path}')
        return self.prediction_tif_path, self.uncertainty_tif_path

    def plot_predictions(self):
        if self.rk_pred is None:
            raise ValueError('Prediction data has been cleared. Call plot_predictions() before generate() completes.')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        extent = [
            float(self.x_grid.min()), float(self.x_grid.max()),
            float(self.y_grid.min()), float(self.y_grid.max()),
        ]
        im = ax.imshow(self.rk_pred, origin='lower', extent=extent, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Depth')
        ax.set(
            xlabel='Easting',
            ylabel='Northing',
            title='Regression Kriging Prediction',
        )
        ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.show()

    def plot_variance(self):
        if self.rk_var is None:
            raise ValueError('Variance data has been cleared. Call plot_variance() before generate() completes.')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        extent = [
            float(self.x_grid.min()), float(self.x_grid.max()),
            float(self.y_grid.min()), float(self.y_grid.max()),
        ]
        im = ax.imshow(self.rk_var, origin='lower', extent=extent, cmap='magma')
        plt.colorbar(im, ax=ax, label='Residual Kriging Variance')
        ax.set(
            xlabel='Easting',
            ylabel='Northing',
            title='Regression Kriging Uncertainty',
        )
        ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.show()

    def plot_variogram(self):
        if self.fit_model is None:
            raise ValueError('No variogram model has been fitted yet. Call generate() first.')

        if not hasattr(self, '_empirical_bins') or not hasattr(self, '_empirical_gamma'):
            raise ValueError('Empirical variogram data not available. Call generate() first.')

        fig, (ax_cloud, ax_fit) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

        # Compute residual variogram cloud.
        x = self.coordinates[0]
        y = self.coordinates[1]
        reg_pred = self.reg_model.predict(
            np.column_stack((self.coordinates[0], self.coordinates[1])).astype(np.float64)
        )
        residuals = self.Z.astype(np.float64) - reg_pred
        n_pts = len(residuals)

        if n_pts > 1:
            max_pairs = min(50000, n_pts * (n_pts - 1) // 2)
            rng = np.random.default_rng(42)
            i = rng.integers(0, n_pts, size=max_pairs)
            j = rng.integers(0, n_pts, size=max_pairs)
            valid = i != j
            i = i[valid]
            j = j[valid]

            h = np.sqrt((x[i].astype(np.float64) - x[j].astype(np.float64)) ** 2 + 
                       (y[i].astype(np.float64) - y[j].astype(np.float64)) ** 2)
            gamma_cloud = 0.5 * (residuals[i] - residuals[j]) ** 2

            ax_cloud.scatter(h, gamma_cloud, s=5, alpha=0.4, c='black', edgecolors='none')
            ax_cloud.set_title('Residual Variogram Cloud', fontsize=12, fontweight='bold')
            ax_cloud.set_xlabel('Lag Distance', fontsize=11)
            ax_cloud.set_ylabel('Semivariance', fontsize=11)
            ax_cloud.grid(True, linestyle='--', alpha=0.3)
        else:
            ax_cloud.text(0.5, 0.5, 'Need at least 2 points for variogram cloud',
                         ha='center', va='center', transform=ax_cloud.transAxes)
            ax_cloud.set_title('Residual Variogram Cloud', fontsize=12, fontweight='bold')

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
        ax_fit.set_title(f'Empirical + Fitted Residual Variogram (AIC={self.fit_aic:.2f})', 
                        fontsize=12, fontweight='bold')
        ax_fit.legend(loc='lower right', fontsize=10)
        ax_fit.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()