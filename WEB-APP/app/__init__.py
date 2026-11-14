import os

from flask import Flask, redirect, render_template, url_for

from app.config import config
from app.logging_config import setup_logging
from app.services.models import ModelService

__version__ = "0.1.0"


def create_app() -> Flask:
    app = Flask(__name__)

    # Load configuration
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    app.config["PROJECT_ROOT"] = config.PROJECT_ROOT

    # Setup logging
    setup_logging(app)

    # Register blueprints
    from app.api.core import core_bp
    from app.api.data_center import data_center_bp

    app.register_blueprint(core_bp)
    app.register_blueprint(data_center_bp)

    # Visualization is optional (requires matplotlib family)
    try:
        from ml.visualization.visualization_api import visualization_bp
        app.register_blueprint(visualization_bp)
    except Exception as e:
        app.logger.warning(f"Visualization blueprint not loaded: {e}")

    # Routes
    @app.route("/")
    def index():
        return render_template("index.html", app_version=__version__)

    @app.route("/visualization")
    def visualization_redirect():
        return redirect(url_for("index"))

    @app.route("/workflow")
    def workflow():
        # Workflow is now integrated into the SPA; keep redirect for bookmarks.
        return render_template("index.html", app_version=__version__)

    # Preload default model lazily on first request if needed
    _preloaded = False

    @app.before_request
    def preload_default_model():
        nonlocal _preloaded
        if not _preloaded and ModelService.active() is None:
            ModelService.ensure_loaded("ET", config.PROJECT_ROOT)
            _preloaded = True

    return app
