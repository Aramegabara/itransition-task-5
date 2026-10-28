import io
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as sps

URL = os.environ.get(
    'SHEET_URL',
    'https://docs.google.com/spreadsheets/d/e/2PACX-1vRLQmQ4tszVZTDqHAAm59xB0iziieQmqevKeiWGbzdOsbOpjvi0rKf-JNZeOIb5ml-HZGkCUiNVM4e1/pub?gid=269770232&single=true&output=csv')

app = Flask(__name__)


def load():
    df = pd.read_csv(URL, usecols=[0, 1, 2], decimal=',', parse_dates=['Date'])
    total = df.groupby('Date', as_index=False)['Production'].sum().assign(Mine='Total')
    return pd.concat([df, total], ignore_index=True)


def stats_table(df):
    g = df.groupby('Mine')['Production']
    return pd.DataFrame({
        'mean': g.mean(),
        'std': g.std(),
        'median': g.median(),
        'iqr': g.quantile(0.75) - g.quantile(0.25),
    }).round(1)


def iqr_test(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return (s < q1 - k * (q3 - q1)) | (s > q3 + k * (q3 - q1))


def z_test(s, z=3.0):
    return ((s - s.mean()) / s.std()).abs() > z


def ma_test(s, window=7, pct=30.0):
    ma = s.rolling(window, center=True, min_periods=1).mean()
    return ((s - ma) / ma * 100).abs() > pct


def grubbs_test(s, alpha=0.05):
    n = len(s)
    t = sps.t.ppf(1 - alpha / (2 * n), n - 2)
    crit = (n - 1) / np.sqrt(n) * np.sqrt(t ** 2 / (n - 2 + t ** 2))
    return ((s - s.mean()) / s.std()).abs() > crit


TESTS = {
    'iqr': ('IQR rule', iqr_test),
    'z': ('Z-score', z_test),
    'ma': ('Moving average deviation', ma_test),
    'grubbs': ('Grubbs test', grubbs_test),
}


def anomalies(series, p):
    return {
        'iqr': iqr_test(series, p['k']),
        'z': z_test(series, p['z']),
        'ma': ma_test(series, p['window'], p['pct']),
        'grubbs': grubbs_test(series, p['alpha']),
    }


def draw(ax, dates, values, kind, degree, flags, title):
    if kind == 'bar':
        ax.bar(dates, values, width=1.0, color='#2563eb', linewidth=0)
    elif kind == 'stacked':
        ax.stackplot(dates, values, colors=['#2563eb'])
    else:
        ax.plot(dates, values, linewidth=1.0, color='#2563eb')

    if flags is not None and flags.any():
        ax.scatter(dates[flags.values], values[flags.values], s=22,
                   color='#dc2626', zorder=5, label='anomaly')

    if degree:
        x = np.arange(len(values))
        coef = np.polyfit(x, values, degree)
        ax.plot(dates, np.polyval(coef, x), linewidth=1.8,
                color='#f59e0b', label=f'trend deg {degree}')

    ax.set_title(title, fontsize=11, loc='left')
    ax.set_ylabel('tons')
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors='#6b7280', labelsize=8)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, frameon=False)


def params_from(src):
    return {
        'mine': src.get('mine', 'Total'),
        'kind': src.get('kind', 'line'),
        'degree': int(src.get('degree', 1)),
        'test': src.get('test', 'iqr'),
        'k': float(src.get('k', 1.5)),
        'z': float(src.get('z', 3)),
        'window': int(src.get('window', 7)),
        'pct': float(src.get('pct', 30)),
        'alpha': float(src.get('alpha', 0.05)),
    }


@app.route('/')
def index():
    p = params_from(request.args)
    df = load()
    mines = sorted(df['Mine'].unique())
    series = df[df['Mine'] == p['mine']].sort_values('Date').reset_index(drop=True)
    flags = anomalies(series['Production'], p)

    st = stats_table(df)
    row = st.loc[p['mine']]
    cards = [('Mean', f"{row['mean']:,.1f}"), ('Std dev', f"{row['std']:,.1f}"),
             ('Median', f"{row['median']:,.1f}"), ('IQR', f"{row['iqr']:,.1f}"),
             ('Anomalies', int(flags[p['test']].sum()))]

    counts = pd.DataFrame({'anomalies': {TESTS[k][0]: int(v.sum()) for k, v in flags.items()}})
    sel = flags[p['test']]
    anom = series[sel.values][['Date', 'Production']].copy()
    anom['Date'] = anom['Date'].dt.strftime('%Y-%m-%d')

    return render_template(
        'index.html', p=p, mines=mines, tests=[(k, v[0]) for k, v in TESTS.items()],
        test_name=TESTS[p['test']][0], qs=request.query_string.decode(),
        cards=cards,
        stats_html=st.to_html(classes='t', border=0),
        counts_html=counts.to_html(border=0),
        anom_html=anom.to_html(index=False, border=0) if len(anom) else '<p>None found.</p>')


@app.route('/chart.png')
def chart():
    p = params_from(request.args)
    df = load()
    s = df[df['Mine'] == p['mine']].sort_values('Date').reset_index(drop=True)
    flags = anomalies(s['Production'], p)[p['test']]
    fig, ax = plt.subplots(figsize=(11, 4))
    draw(ax, s['Date'], s['Production'], p['kind'], p['degree'], flags,
         f"{p['mine']} — {TESTS[p['test']][0]}")
    fig.autofmt_xdate()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


def table_page(pdf, title, frame):
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis('off')
    ax.set_title(title, fontsize=13, loc='left', pad=18)
    data = frame.reset_index() if frame.index.name else frame
    t = ax.table(cellText=data.round(1).astype(str).values if data.select_dtypes('number').shape[1] else data.astype(str).values,
                 colLabels=list(data.columns), loc='upper left', cellLoc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.4)
    pdf.savefig(fig)
    plt.close(fig)


@app.route('/report.pdf')
def report():
    p = params_from(request.args)
    df = load()
    mines = sorted(df['Mine'].unique())
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(0.08, 0.72, 'Weyland-Yutani', fontsize=30, weight='bold')
        fig.text(0.08, 0.65, 'Mining production report', fontsize=19, color='#6b7280')
        fig.text(0.08, 0.55, f"Period: {df['Date'].min():%Y-%m-%d} — {df['Date'].max():%Y-%m-%d}", fontsize=12)
        fig.text(0.08, 0.51, f"Mines: {len(mines) - 1}    Records: {len(df)}", fontsize=12)
        fig.text(0.08, 0.10, 'Artur Heraskov', fontsize=11, color='#6b7280')
        pdf.savefig(fig)
        plt.close(fig)

        table_page(pdf, 'Statistics per mine', stats_table(df))

        for mine in mines:
            s = df[df['Mine'] == mine].sort_values('Date').reset_index(drop=True)
            flags = anomalies(s['Production'], p)
            fig, axes = plt.subplots(3, 1, figsize=(11.7, 8.3))
            for ax, kind in zip(axes, ['line', 'bar', 'stacked']):
                draw(ax, s['Date'], s['Production'], kind, p['degree'],
                     flags[p['test']], f'{mine} — {kind}')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            for key, (name, _) in TESTS.items():
                sel = flags[key]
                fig, ax = plt.subplots(figsize=(11.7, 8.3))
                ax.axis('off')
                ax.set_title(f'{mine} — {name}: {int(sel.sum())} anomalies',
                             fontsize=13, loc='left', pad=16)
                rows = s[sel.values][['Date', 'Production']].head(40)
                text = '\n'.join(f"{d:%Y-%m-%d}    {v:>10,.1f}"
                                 for d, v in zip(rows['Date'], rows['Production']))
                ax.text(0.02, 0.95, text or 'None found.', fontsize=9,
                        family='monospace', va='top')
                pdf.savefig(fig)
                plt.close(fig)

    buf.seek(0)
    return send_file(buf, mimetype='application/pdf',
                     download_name='weyland_yutani_report.pdf')


if __name__ == '__main__':
    app.run(debug=True, port=5000)