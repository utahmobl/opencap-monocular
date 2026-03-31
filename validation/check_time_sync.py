import os
import webbrowser
import argparse
from pathlib import Path


def find_reports_generator(root_folder):
    """
    Walks through a directory structure and yields the paths of lag correlation HTML reports.
    This is memory efficient as it doesn't load all paths at once.
    """
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.startswith("lag_correlation") and filename.endswith(".html"):
                yield os.path.join(dirpath, filename)


def create_summary_page(report_paths, output_filepath):
    """
    Creates an HTML summary page with links to all found reports.
    """
    with open(output_filepath, "w") as f:
        f.write("<!DOCTYPE html>\n")
        f.write('<html lang="en">\n')
        f.write("<head>\n")
        f.write('    <meta charset="UTF-8">\n')
        f.write("    <title>Lag Correlation Reports Summary</title>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        f.write("    <h1>Lag Correlation Reports</h1>\n")
        f.write("    <ul>\n")

        for path in report_paths:
            uri = Path(os.path.abspath(path)).as_uri()
            f.write(f'        <li><a href="{uri}" target="_blank">{path}</a></li>\n')

        f.write("    </ul>\n")
        f.write("</body>\n")
        f.write("</html>\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find lag correlation HTML reports and generates a summary page.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="One or more paths, which can be root folders to search, specific HTML report files, or .txt files containing a list of report files.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the summary page in a browser automatically.",
    )
    args = parser.parse_args()

    reports = []
    for path in args.paths:
        if os.path.isdir(path):
            reports.extend(find_reports_generator(path))
        elif os.path.isfile(path):
            if path.endswith(".txt"):
                try:
                    with open(path, "r") as f:
                        for line in f:
                            report_path = line.strip()
                            if report_path:
                                reports.append(report_path)
                except IOError as e:
                    print(
                        f"Warning: Could not read file '{path}'. Error: {e}. Skipping."
                    )
            elif "lag_correlation" in os.path.basename(path) and path.endswith(".html"):
                reports.append(path)
            else:
                print(
                    f"Warning: Provided file '{path}' is not a lag correlation report (.html) or a list of reports (.txt). Skipping."
                )
        else:
            print(f"Warning: '{path}' is not a valid file or directory. Skipping.")

    if reports:
        # Remove duplicates and sort
        reports = sorted(list(set(reports)))
        summary_filename = "lag_correlation_summary.html"
        create_summary_page(reports, summary_filename)
        summary_path = os.path.abspath(summary_filename)
        print(f"Summary page created at: {summary_path}")

        if not args.no_open:
            webbrowser.open_new_tab(f"file://{summary_path}")
    else:
        print("No 'lag_correlation*.html' files found or valid files provided.")
