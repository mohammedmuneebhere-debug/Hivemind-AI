from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = ROOT / "demo_episode.py"
t = p.read_text(encoding="utf-8")
if "--mp4-out" in t:
    print("demo_episode already patched")
    raise SystemExit(0)
t = t.replace(
    'parser.add_argument("--gif-out", type=str, default="logs/ui/demo_motion.gif")\n'
    '    parser.add_argument("--trace-out"',
    'parser.add_argument("--gif-out", type=str, default="logs/ui/demo_motion.gif")\n'
    '    parser.add_argument("--mp4-out", type=str, default="logs/ui/demo_motion.mp4")\n'
    '    parser.add_argument("--trace-out"',
)
t = t.replace(
    "    gif_out = Path(args.gif_out)\n    trace_out = Path(args.trace_out)",
    "    gif_out = Path(args.gif_out)\n    mp4_out = Path(args.mp4_out)\n    trace_out = Path(args.trace_out)",
)
t = t.replace(
    "from matplotlib.animation import FuncAnimation, PillowWriter",
    "from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter",
)
t = t.replace(
    '    anim.save(str(gif_out), writer=PillowWriter(fps=5))\n    plt.close(fig)\n\n    print(f"Wrote GIF: {gif_out}")',
    '    anim.save(str(gif_out), writer=PillowWriter(fps=5))\n    try:\n'
    '        anim.save(str(mp4_out), writer=FFMpegWriter(fps=10))\n'
    '        print(f"Wrote MP4: {mp4_out}")\n'
    "    except Exception as e:\n"
    '        print(f"MP4 skipped ({e})")\n'
    '    plt.close(fig)\n\n    print(f"Wrote GIF: {gif_out}")',
)
p.write_text(t, encoding="utf-8")
print("patched", p)
