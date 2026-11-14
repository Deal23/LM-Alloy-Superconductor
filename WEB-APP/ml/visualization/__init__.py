import io
import base64
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import inspect
import threading

_plot_lock = threading.RLock()

from .plot_generator import (
    plot_fig2a as _plot_fig2a,
    plot_fig2b as _plot_fig2b,
    plot_fig2c as _plot_fig2c,
)


def _generate_plot(plot_func, *args, **kwargs):
    # Matplotlib has global state. Serialize rendering and suppress legacy disk
    # exports: web previews live in memory; only browser downloads persist.
    with _plot_lock:
        original_savefig = plt.savefig
        plt.savefig = lambda *a, **kw: None
        try:
            return _render_plot(plot_func, *args, **kwargs)
        finally:
            plt.savefig = original_savefig


def _render_plot(plot_func, *args, **kwargs):
    dpi = kwargs.pop("dpi", 300)
    return_base64 = kwargs.pop("return_base64", True)

    # Filter kwargs to only those accepted by the original function
    sig = inspect.signature(plot_func)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    # Mock plt.show to prevent blocking or clearing the figure
    original_show = plt.show
    plt.show = lambda: None

    try:
        plot_func(*args, **valid_kwargs)

        if return_base64:
            buf = io.BytesIO()
            # Capture the current figure
            fig = plt.gcf()
            fig.savefig(
                buf, format="png", dpi=dpi, bbox_inches="tight", transparent=True
            )
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close("all")
            return {"success": True, "image": img_base64}
        else:
            plt.close("all")
            return {"success": True}
    except Exception as e:
        import traceback

        plt.close("all")
        return {"success": False, "error": str(e) + "\n" + traceback.format_exc()}
    finally:
        plt.show = original_show


def generate_fig2a(**kwargs):
    return _generate_plot(_plot_fig2a, **kwargs)


def generate_fig2b(**kwargs):
    return _generate_plot(_plot_fig2b, **kwargs)


def generate_fig2c(**kwargs):
    return _generate_plot(_plot_fig2c, **kwargs)


def generate_fig4b(**kwargs):
    return {"success": False, "error": "历史数据收敛曲线不适用于当前基础数据；请重新计算。"}


def generate_fig4c(**kwargs):
    return {"success": False, "error": "旧二元合金结果不代表当前模型，请先重新筛选并使用筛选结果。"}


def generate_fig5(**kwargs):
    return {"success": False, "error": "当前没有保存独立测试集预测，不能生成预测与实验对比图。"}


def generate_fig5d(**kwargs):
    return {"success": False, "error": "历史模型对比数据不适用于当前基础数据。"}


def generate_ternary(*args, **kwargs):
    """Stub - ternary plot not yet implemented"""
    return {"success": False, "error": "Ternary plot generation not available yet"}


def generate_all_plots(*args, **kwargs):
    """Generate all plots at once"""
    return {
        "fig2a": generate_fig2a(**kwargs),
        "fig2b": generate_fig2b(**kwargs),
        "fig2c": generate_fig2c(**kwargs),
        "fig4b": generate_fig4b(**kwargs),
        "fig4c": generate_fig4c(**kwargs),
        "fig5": generate_fig5(**kwargs),
        "fig5d": generate_fig5d(**kwargs),
        "ternary": generate_ternary(**kwargs),
    }


__all__ = [
    "generate_fig2a",
    "generate_fig2b",
    "generate_fig2c",
    "generate_fig4b",
    "generate_fig4c",
    "generate_fig5",
    "generate_fig5d",
    "generate_ternary",
    "generate_all_plots",
]
