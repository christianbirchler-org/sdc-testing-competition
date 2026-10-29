{ pkgs, lib, config, inputs, ... }:

{
    # https://devenv.sh/basics/
    env.GREET = "devenv";

    # https://devenv.sh/packages/
    packages = [
    pkgs.pyright
    pkgs.black
    ];

    # https://devenv.sh/languages/
    languages.python = {
      enable = true;
      version = "3.13.17";
      venv.enable = true;
    };

    # https://devenv.sh/services/
    services.postgres = {
        enable = true;
        initialDatabases = [{
            name = "sdc-tool-competition-2027-evaluation";
        }];
    };

    # https://devenv.sh/scripts/
    scripts.hello.exec = ''
      echo hello from $GREET
    '';

    # https://devenv.sh/basics/
    enterShell = ''
      hello         # Run scripts directly
      git --version # Use packages
    '';

    # https://devenv.sh/tests/
    enterTest = ''
      echo "Running tests"
      git --version | grep --color=auto "${pkgs.git.version}"
    '';
}
