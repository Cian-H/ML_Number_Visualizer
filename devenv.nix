{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  languages.python = {
    enable = true;
    version = "3.14";
    venv.enable = true;
    uv.enable = true;
    uv.sync.enable = true;
  };
}
