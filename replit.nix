
{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.flask
    pkgs.python311Packages.pandas
    pkgs.python311Packages.numpy
    pkgs.python311Packages.plotly
    pkgs.python311Packages.gunicorn
    pkgs.python311Packages.setuptools
    pkgs.python311Packages.yfinance
  ];
}
