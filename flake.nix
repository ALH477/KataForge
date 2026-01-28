{
  description = "KataForge v.1 - Martial Arts Coach Preservation & Training System";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix = {
  url = "github:nix-community/poetry2nix/master";
  inputs.nixpkgs.follows = "nixpkgs";
};
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgsRocm = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            rocmSupport = true;
            cudaSupport = false;
          };
        };

        pkgsCuda = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            rocmSupport = false;
          };
        };

        pkgsVulkan = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };

        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };

        p2nix = poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };
        p2nixRocm = poetry2nix.lib.mkPoetry2Nix { pkgs = pkgsRocm; };
        p2nixCuda = poetry2nix.lib.mkPoetry2Nix { pkgs = pkgsCuda; };
        p2nixVulkan = poetry2nix.lib.mkPoetry2Nix { pkgs = pkgsVulkan; };

        pythonVersion = "python311";

        nativeBuildDeps = pkgs: with pkgs; [
          pkg-config
          cmake
          gcc
          ffmpeg
          libGL
          libGLU
          xorg.libX11
          xorg.libXext
          xorg.libXrender
        ];

        runtimeDeps = pkgs: with pkgs; [
          ffmpeg
          libGL
        ];

        vulkanDeps = pkgSet: with pkgSet; [
          vulkan-headers
          vulkan-loader
          vulkan-tools
          vulkan-validation-layers
          shaderc
        ];

        commonOverrides = pkgSet: self: super: {
          opencv-python = super.opencv-python.overridePythonAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ (nativeBuildDeps pkgSet);
            buildInputs = (old.buildInputs or []) ++ (runtimeDeps pkgSet);
            dontUseCmakeConfigure = true;
          });
          
          mediapipe = super.mediapipe.overridePythonAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ pkgSet.protobuf ];
            buildInputs = (old.buildInputs or []) ++ (runtimeDeps pkgSet);
          });
          
          torch-geometric = super.torch-geometric.overridePythonAttrs (old: {
            buildInputs = (old.buildInputs or []) ++ [ self.torch ];
          });
        };

        rocmPackages = with pkgsRocm.rocmPackages; [
          rocm-smi
          rocminfo
          rocm-runtime
          rocm-device-libs
          clr
        ];

        rocmOverrides = p2nixRocm.overrides.withDefaults (self: super:
          (commonOverrides pkgsRocm self super) // {
            torch = super.torch.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or []) ++ rocmPackages;
            });
          }
        );

        mkRocmPythonEnv = p2nixRocm.mkPoetryEnv {
          projectDir = ./.;
          python = pkgsRocm.${pythonVersion};
          preferWheels = true;
          overrides = rocmOverrides;
          groups = [ ];
          checkGroups = [ ];
          extras = [ ];
        };

        cudaDeps = with pkgsCuda.cudaPackages; [
          cudatoolkit
          cudnn
        ];

        cudaOverrides = p2nixCuda.overrides.withDefaults (self: super:
          (commonOverrides pkgsCuda self super) // {
            torch = super.torch.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or []) ++ cudaDeps;
            });
          }
        );

        mkCudaPythonEnv = p2nixCuda.mkPoetryEnv {
          projectDir = ./.;
          python = pkgsCuda.${pythonVersion};
          preferWheels = true;
          overrides = cudaOverrides;
          groups = [ ];
          checkGroups = [ ];
          extras = [ ];
        };

        vulkanOverrides = p2nixVulkan.overrides.withDefaults (self: super:
          (commonOverrides pkgsVulkan self super) // {
            torch = super.torch;
          }
        );

        mkVulkanPythonEnv = p2nixVulkan.mkPoetryEnv {
          projectDir = ./.;
          python = pkgsVulkan.${pythonVersion};
          preferWheels = true;
          overrides = vulkanOverrides;
          groups = [ ];
          checkGroups = [ ];
          extras = [ ];
        };

        cpuOverrides = p2nix.overrides.withDefaults (self: super:
          commonOverrides pkgs self super
        );

        mkCpuPythonEnv = p2nix.mkPoetryEnv {
          projectDir = ./.;
          python = pkgs.${pythonVersion};
          preferWheels = true;
          overrides = cpuOverrides;
          groups = [ ];
          checkGroups = [ ];
          extras = [ ];
        };

        llamaCppVulkan = pkgsVulkan.stdenv.mkDerivation rec {
          pname = "llama-cpp-vulkan";
          version = "b4292";
          
          src = pkgsVulkan.fetchFromGitHub {
            owner = "ggerganov";
            repo = "llama.cpp";
            rev = version;
            sha256 = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
          };
          
          nativeBuildInputs = with pkgsVulkan; [
            cmake
            pkg-config
            git
          ];
          
          buildInputs = with pkgsVulkan; [
            vulkan-headers
            vulkan-loader
            shaderc
            glslang
          ];
          
          cmakeFlags = [
            "-DGGML_VULKAN=ON"
            "-DLLAMA_CURL=OFF"
            "-DLLAMA_BUILD_TESTS=OFF"
            "-DLLAMA_BUILD_EXAMPLES=ON"
            "-DLLAMA_BUILD_SERVER=ON"
          ];
          
          installPhase = ''
            mkdir -p $out/bin
            cp bin/llama-server $out/bin/
            cp bin/llama-cli $out/bin/
            cp bin/llama-quantize $out/bin/
          '';
          
          meta = with pkgsVulkan.lib; {
            description = "llama.cpp with Vulkan backend for portable GPU inference";
            homepage = "https://github.com/ggerganov/llama.cpp";
            license = licenses.mit;
            platforms = platforms.linux;
          };
        };

        ggufModels = pkgs.stdenv.mkDerivation {
          pname = "kataforge-gguf-models";
          version = "1.0.0";
          
          phases = [ "installPhase" ];
          
          installPhase = ''
            mkdir -p $out/models
            echo "Models would be placed here" > $out/models/README.txt
          '';
          
          meta = with pkgs.lib; {
            description = "Pre-bundled GGUF models for KataForge v.1";
            license = licenses.unfree;
          };
        };

        mkDojoManagerApp = pythonEnv: pkgSet: pkgSet.stdenv.mkDerivation {
          pname = "kataforge";
          version = "0.1.0";
          
          src = ./.;
          
          buildInputs = [ pythonEnv ] ++ (runtimeDeps pkgSet);
          
          installPhase = ''
            mkdir -p $out/bin $out/lib/kataforge
            cp -r kataforge $out/lib/kataforge/
            cp pyproject.toml $out/lib/kataforge/
            
            cat > $out/bin/kataforge << EOF
            #!${pkgSet.bash}/bin/bash
            export PYTHONPATH="$out/lib/kataforge:\$PYTHONPATH"
            exec ${pythonEnv}/bin/python -m kataforge.cli.main "\$@"
            EOF
            chmod +x $out/bin/kataforge
            
            cat > $out/bin/kataforge-server << EOF
            #!${pkgSet.bash}/bin/bash
            export PYTHONPATH="$out/lib/kataforge:\$PYTHONPATH"
            exec ${pythonEnv}/bin/python -m kataforge.api.server "\$@"
            EOF
            chmod +x $out/bin/kataforge-server
          '';
          
          meta = with pkgSet.lib; {
            description = "Martial Arts Coach Preservation & Training System";
            homepage = "https://github.com/demod-llc/kataforge";
            license = licenses.unfree;
            platforms = platforms.linux ++ platforms.darwin;
          };
        };

        mkDojoGradioApp = pythonEnv: pkgSet: pkgSet.stdenv.mkDerivation {
          pname = "kataforge-ui";
          version = "0.1.0";
          
          src = ./.;
          
          buildInputs = [ pythonEnv ] ++ (runtimeDeps pkgSet);
          
          installPhase = ''
            mkdir -p $out/bin $out/lib/kataforge
            cp -r kataforge $out/lib/kataforge/
            cp pyproject.toml $out/lib/kataforge/
            
            cat > $out/bin/kataforge-ui << EOF
            #!${pkgSet.bash}/bin/bash
            export PYTHONPATH="$out/lib/kataforge:\$PYTHONPATH"
            exec ${pythonEnv}/bin/python -m kataforge.cli.main ui "\$@"
            EOF
            chmod +x $out/bin/kataforge-ui
          '';
          
          meta = with pkgSet.lib; {
            description = "KataForge v.1 Gradio Web Interface";
            homepage = "https://github.com/demod-llc/kataforge";
            license = licenses.unfree;
            platforms = platforms.linux ++ platforms.darwin;
          };
        };

        commonDevTools = with pkgs; [
          poetry
          git
          ruff
          mypy
          htop
          jq
          curl
        ];

        rocmShell = pkgsRocm.mkShell {
          name = "kataforge-rocm";
          
          buildInputs = [
            mkRocmPythonEnv
          ] ++ rocmPackages ++ commonDevTools ++ (nativeBuildDeps pkgsRocm) ++ [
            pkgsRocm.nvtopPackages.amd
          ];

          shellHook = ''
            export ROCM_PATH=${pkgsRocm.rocmPackages.clr}
            export HIP_PATH=${pkgsRocm.rocmPackages.clr}
            export HSA_OVERRIDE_GFX_VERSION=''${HSA_OVERRIDE_GFX_VERSION:-11.0.0}
            export PYTORCH_ROCM_ARCH=''${PYTORCH_ROCM_ARCH:-gfx1100}
            export HSA_ENABLE_SDMA=0
            export GPU_MAX_HEAP_SIZE=100
            export GPU_MAX_ALLOC_PERCENT=100
            export LD_LIBRARY_PATH=${pkgsRocm.rocmPackages.clr}/lib:''${LD_LIBRARY_PATH:-}
            export PYTHONPATH="$PWD:''${PYTHONPATH:-}"

            echo ""
            echo "╔═╦═════════════ ════════════ ════════════════════════════════╗"
            echo "║  KataForge v.1 - ROCm/AMD GPU Development Environment          ║"
            echo "╠═╦═══════════════════════════════ ════════════════╦══════════╣"
            echo "║  GPU Backend: ROCm (AMD)                                      ║"
            echo "║  Python: $(python --version 2>&1)                             ║"
            echo "║                                                               ║"
            echo "║  Commands:                                                    ║"
            echo "║    kataforge --help   CLI help                                ║"
            echo "║    kataforge serve    Start API server                        ║"
            echo "║    rocm-smi           Check GPU status                        ║"
            echo "║    nvtop              GPU monitoring                          ║"
            echo "╚═══════════════════════════════ ════════════════╝"
            echo ""

            python -c "import torch; print(f'PyTorch: {torch.__version__}, ROCm available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" 2>/dev/null || echo "Note: Run 'poetry install' if packages are missing"
          '';
        };

        cudaShell = pkgsCuda.mkShell {
          name = "kataforge-cuda";
          
          buildInputs = [
            mkCudaPythonEnv
          ] ++ cudaDeps ++ commonDevTools ++ (nativeBuildDeps pkgsCuda) ++ [
            pkgsCuda.nvtopPackages.nvidia
          ];

          shellHook = ''
            export CUDA_HOME=${pkgsCuda.cudaPackages.cudatoolkit}
            export LD_LIBRARY_PATH=${pkgsCuda.cudaPackages.cudatoolkit}/lib:''${LD_LIBRARY_PATH:-}
            export PYTHONPATH="$PWD:''${PYTHONPATH:-}"

            echo ""
            echo "╔═══════════════════════════════════════════════════════════════╗"
            echo "║  KataForge v.1 - CUDA/NVIDIA GPU Development Environment       ║"
            echo "╠═══════════════════════════════════════════════════════════════╣"
            echo "║  GPU Backend: CUDA (NVIDIA)                                   ║"
            echo "║  Python: $(python --version 2>&1)                             ║"
            echo "║                                                               ║"
            echo "║  Commands:                                                    ║"
            echo "║    kataforge --help   CLI help                                ║"
            echo "║    kataforge serve    Start API server                        ║"
            echo "║    nvidia-smi         Check GPU status                        ║"
            echo "║    nvtop              GPU monitoring                          ║"
            echo "╚═══════════════════════════════════════════════════════════════╝"
            echo ""

            python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" 2>/dev/null || echo "Note: Run 'poetry install' if packages are missing"
          '';
        };

        vulkanShell = pkgsVulkan.mkShell {
          name = "kataforge-vulkan";
          
          buildInputs = [
            mkVulkanPythonEnv
          ] ++ (vulkanDeps pkgsVulkan) ++ commonDevTools ++ (nativeBuildDeps pkgsVulkan);

          shellHook = ''
            export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json:/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
            export PYTHONPATH="$PWD:''${PYTHONPATH:-}"

            echo ""
            echo "╔═══════════════════════════════════════════════════════════════╗"
            echo "║  KataForge v.1 - Vulkan Development Environment                ║"
            echo "╠═══════════════════════════════════════════════════════════════╣"
            echo "║  GPU Backend: Vulkan (Intel/AMD/portable)                     ║"
            echo "║  Python: $(python --version 2>&1)                             ║"
            echo "║  LLM: llama.cpp with Vulkan backend                           ║"
            echo "║                                                               ║"
            echo "║  Commands:                                                    ║"
            echo "║    kataforge --help   CLI help                                ║"
            echo "║    kataforge serve    Start API server                        ║"
            echo "║    vulkaninfo         Check Vulkan status                     ║"
            echo "╚═══════════════════════════════════════════════════════════════╝"
            echo ""

            python -c "import torch; print(f'PyTorch: {torch.__version__} (CPU - Vulkan used for LLM)')" 2>/dev/null || echo "Note: Run 'poetry install' if packages are missing"
          '';
        };

        cpuShell = pkgs.mkShell {
          name = "kataforge-cpu";
          
          buildInputs = [
            mkCpuPythonEnv
          ] ++ commonDevTools ++ (nativeBuildDeps pkgs);

          shellHook = ''
            export PYTHONPATH="$PWD:''${PYTHONPATH:-}"

            echo ""
            echo "╔═══════════════════════════════════════════════════════════════╗"
            echo "║  KataForge v.1 - CPU Development Environment                   ║"
            echo "╠═══════════════════════════════════════════════════════════════╣"
            echo "║  GPU Backend: None (CPU only)                                 ║"
            echo "║  Python: $(python --version 2>&1)                             ║"
            echo "║                                                               ║"
            echo "║  For GPU support, use:                                        ║"
            echo "║    nix develop .#rocm   - AMD ROCm                            ║"
            echo "║    nix develop .#cuda   - NVIDIA CUDA                         ║"
            echo "║    nix develop .#vulkan - Vulkan (Intel/AMD)                  ║"
            echo "║                                                               ║"
            echo "║  Commands:                                                    ║"
            echo "║    kataforge --help   CLI help                                ║"
            echo "║    kataforge serve    Start API server                        ║"
            echo "╚═══════════════════════════════════════════════════════════════╝"
            echo ""

            python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "Note: Run 'poetry install' if packages are missing"
          '';
        };

        rocmDockerImage = pkgsRocm.dockerTools.buildLayeredImage {
          name = "kataforge";
          tag = "rocm";
          
          contents = [
            (mkDojoManagerApp mkRocmPythonEnv pkgsRocm)
            mkRocmPythonEnv
            pkgsRocm.bash
            pkgsRocm.coreutils
            pkgsRocm.cacert
            pkgsRocm.curl
          ];
          
          config = {
            Env = [
              "PATH=/bin:${mkRocmPythonEnv}/bin"
              "PYTHONPATH=/lib/kataforge"
              "SSL_CERT_FILE=${pkgsRocm.cacert}/etc/ssl/certs/ca-bundle.crt"
              "DOJO_ENVIRONMENT=production"
              "DOJO_LOG_FORMAT=json"
              "DOJO_API_HOST=0.0.0.0"
              "DOJO_API_PORT=8000"
              "DOJO_LLM_BACKEND=ollama"
            ];
            
            WorkingDir = "/app";
            
            Cmd = [ "${mkRocmPythonEnv}/bin/python" "-m" "kataforge.api.server" ];
            
            ExposedPorts = {
              "8000/tcp" = {};
            };
            
            Healthcheck = {
              Test = [ "CMD" "${pkgsRocm.curl}/bin/curl" "-f" "http://localhost:8000/health/live" ];
              Interval = 30000000000;
              Timeout = 10000000000;
              Retries = 3;
              StartPeriod = 60000000000;
            };
            
            Labels = {
              "org.opencontainers.image.title" = "KataForge v.1";
              "org.opencontainers.image.description" = "Martial Arts Technique Analysis with AMD ROCm support";
              "org.opencontainers.image.version" = "0.1.0";
              "org.opencontainers.image.vendor" = "DeMoD LLC";
              "com.demod.dojo.gpu" = "rocm";
            };
          };
        };

        cudaDockerImage = pkgsCuda.dockerTools.buildLayeredImage {
          name = "kataforge";
          tag = "cuda";
          
          contents = [
            (mkDojoManagerApp mkCudaPythonEnv pkgsCuda)
            mkCudaPythonEnv
            pkgsCuda.bash
            pkgsCuda.coreutils
            pkgsCuda.cacert
            pkgsCuda.curl
          ];
          
          config = {
            Env = [
              "PATH=/bin:${mkCudaPythonEnv}/bin"
              "PYTHONPATH=/lib/kataforge"
              "SSL_CERT_FILE=${pkgsCuda.cacert}/etc/ssl/certs/ca-bundle.crt"
              "DOJO_ENVIRONMENT=production"
              "DOJO_LOG_FORMAT=json"
              "DOJO_API_HOST=0.0.0.0"
              "DOJO_API_PORT=8000"
              "DOJO_LLM_BACKEND=ollama"
            ];
            
            WorkingDir = "/app";
            
            Cmd = [ "${mkCudaPythonEnv}/bin/python" "-m" "kataforge.api.server" ];
            
            ExposedPorts = {
              "8000/tcp" = {};
            };
            
            Healthcheck = {
              Test = [ "CMD" "${pkgsCuda.curl}/bin/curl" "-f" "http://localhost:8000/health/live" ];
              Interval = 30000000000;
              Timeout = 10000000000;
              Retries = 3;
              StartPeriod = 60000000000;
            };
            
            Labels = {
              "org.opencontainers.image.title" = "KataForge v.1";
              "org.opencontainers.image.description" = "Martial Arts Technique Analysis with NVIDIA CUDA support";
              "org.opencontainers.image.version" = "0.1.0";
              "org.opencontainers.image.vendor" = "DeMoD LLC";
              "com.demod.dojo.gpu" = "cuda";
            };
          };
        };

        vulkanDockerImage = pkgsVulkan.dockerTools.buildLayeredImage {
          name = "kataforge";
          tag = "vulkan";
          
          contents = [
            (mkDojoManagerApp mkVulkanPythonEnv pkgsVulkan)
            mkVulkanPythonEnv
            pkgsVulkan.bash
            pkgsVulkan.coreutils
            pkgsVulkan.cacert
            pkgsVulkan.curl
          ];
          
          config = {
            Env = [
              "PATH=/bin:${mkVulkanPythonEnv}/bin"
              "PYTHONPATH=/lib/kataforge"
              "SSL_CERT_FILE=${pkgsVulkan.cacert}/etc/ssl/certs/ca-bundle.crt"
              "DOJO_ENVIRONMENT=production"
              "DOJO_LOG_FORMAT=json"
              "DOJO_API_HOST=0.0.0.0"
              "DOJO_API_PORT=8000"
              "DOJO_LLM_BACKEND=llamacpp"
            ];
            
            WorkingDir = "/app";
            
            Cmd = [ "${mkVulkanPythonEnv}/bin/python" "-m" "kataforge.api.server" ];
            
            ExposedPorts = {
              "8000/tcp" = {};
            };
            
            Healthcheck = {
              Test = [ "CMD" "${pkgsVulkan.curl}/bin/curl" "-f" "http://localhost:8000/health/live" ];
              Interval = 30000000000;
              Timeout = 10000000000;
              Retries = 3;
              StartPeriod = 60000000000;
            };
            
            Labels = {
              "org.opencontainers.image.title" = "KataForge v.1";
              "org.opencontainers.image.description" = "Martial Arts Technique Analysis with Vulkan GPU support";
              "org.opencontainers.image.version" = "0.1.0";
              "org.opencontainers.image.vendor" = "DeMoD LLC";
              "com.demod.dojo.gpu" = "vulkan";
            };
          };
        };

        cpuDockerImage = pkgs.dockerTools.buildLayeredImage {
          name = "kataforge";
          tag = "cpu";
          
          contents = [
            (mkDojoManagerApp mkCpuPythonEnv pkgs)
            mkCpuPythonEnv
            pkgs.bash
            pkgs.coreutils
            pkgs.cacert
            pkgs.curl
          ];
          
          config = {
            Env = [
              "PATH=/bin:${mkCpuPythonEnv}/bin"
              "PYTHONPATH=/lib/kataforge"
              "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              "DOJO_ENVIRONMENT=production"
              "DOJO_LOG_FORMAT=json"
              "DOJO_API_HOST=0.0.0.0"
              "DOJO_API_PORT=8000"
            ];
            
            WorkingDir = "/app";
            
            Cmd = [ "${mkCpuPythonEnv}/bin/python" "-m" "kataforge.api.server" ];
            
            ExposedPorts = {
              "8000/tcp" = {};
            };
            
            Healthcheck = {
              Test = [ "CMD" "${pkgs.curl}/bin/curl" "-f" "http://localhost:8000/health/live" ];
              Interval = 30000000000;
              Timeout = 10000000000;
              Retries = 3;
              StartPeriod = 60000000000;
            };
            
            Labels = {
              "org.opencontainers.image.title" = "KataForge v.1";
              "org.opencontainers.image.description" = "Martial Arts Technique Analysis (CPU only)";
              "org.opencontainers.image.version" = "0.1.0";
              "org.opencontainers.image.vendor" = "DeMoD LLC";
              "com.demod.dojo.gpu" = "none";
            };
          };
        };

        rocmGradioDockerImage = pkgsRocm.dockerTools.buildLayeredImage {
          name = "kataforge-ui";
          tag = "rocm";
          
          contents = [
            (mkDojoGradioApp mkRocmPythonEnv pkgsRocm)
            mkRocmPythonEnv
            pkgsRocm.bash
            pkgsRocm.coreutils
            pkgsRocm.cacert
            pkgsRocm.curl
          ];
          
          config = {
            Env = [
              "PATH=/bin:${mkRocmPythonEnv}/bin"
              "PYTHONPATH=/lib/kataforge"
              "SSL_CERT_FILE=${pkgsRocm.cacert}/etc/ssl/certs/ca-bundle.crt"
              "GRADIO_SERVER_NAME=0.0.0.0"
              "GRADIO_SERVER_PORT=7860"
              "DOJO_LLM_BACKEND=ollama"
            ];
            
            WorkingDir = "/app";
            
            Cmd = [ "${mkRocmPythonEnv}/bin/python" "-m" "kataforge.cli.main" "ui" ];
            
            ExposedPorts = {
              "7860/tcp" = {};
            };
            
            Healthcheck = {
              Test = [ "CMD" "${pkgsRocm.curl}/bin/curl" "-f" "http://localhost:7860/" ];
              Interval = 30000000000;
              Timeout = 10000000000;
              Retries = 3;
              StartPeriod = 60000000000;
            };
            
            Labels = {
              "org.opencontainers.image.title" = "KataForge v.1 UI";
              "org.opencontainers.image.description" = "KataForge v.1 Gradio UI with AMD ROCm support";
              "org.opencontainers.image.version" = "0.1.0";
              "org.opencontainers.image.vendor" = "DeMoD LLC";
              "com.demod.dojo.component" = "gradio-ui";
            };
          };
        };

        cudaGradioDockerImage = pkgsCuda.dockerTools.buildLayeredImage {
          name = "kataforge-ui";
          tag = "cuda";
          
          contents = [
            (mkDojoGradioApp mkCudaPythonEnv pkgsCuda)
            mkCudaPythonEnv
            pkgsCuda.bash
            pkgsCuda.coreutils
            pkgsCuda.cacert
            pkgsCuda.curl
          ];
          
          config = {
            Env = [
              "PATH=/bin:${mkCudaPythonEnv}/bin"
              "PYTHONPATH=/lib/kataforge"
              "SSL_CERT_FILE=${pkgsCuda.cacert}/etc/ssl/certs/ca-bundle.crt"
              "GRADIO_SERVER_NAME=0.0.0.0"
              "GRADIO_SERVER_PORT=7860"
              "DOJO_LLM_BACKEND=ollama"
            ];
            
            WorkingDir = "/app";
            
            Cmd = [ "${mkCudaPythonEnv}/bin/python" "-m" "kataforge.cli.main" "ui" ];
            
            ExposedPorts = {
              "7860/tcp" = {};
            };
            
            Healthcheck = {
              Test = [ "CMD" "${pkgsCuda.curl}/bin/curl" "-f" "http://localhost:7860/" ];
              Interval = 30000000000;
              Timeout = 10000000000;
              Retries = 3;
              StartPeriod = 60000000000;
            };
            
            Labels = {
              "org.opencontainers.image.title" = "KataForge v.1 UI";
              "org.opencontainers.image.description" = "KataForge v.1 Gradio UI with NVIDIA CUDA support";
              "org.opencontainers.image.version" = "0.1.0";
              "org.opencontainers.image.vendor" = "DeMoD LLC";
              "com.demod.dojo.component" = "gradio-ui";
            };
          };
        };

        vulkanGradioDockerImage = pkgsVulkan.dockerTools.buildLayeredImage {
          name = "kataforge-ui";
          tag = "vulkan";
          
          contents = [
            (mkDojoGradioApp mkVulkanPythonEnv pkgsVulkan)
            mkVulkanPythonEnv
            pkgsVulkan.bash
            pkgsVulkan.coreutils
            pkgsVulkan.cacert
            pkgsVulkan.curl
          ];
          
          config = {
            Env = [
              "PATH=/bin:${mkVulkanPythonEnv}/bin"
              "PYTHONPATH=/lib/kataforge"
              "SSL_CERT_FILE=${pkgsVulkan.cacert}/etc/ssl/certs/ca-bundle.crt"
              "GRADIO_SERVER_NAME=0.0.0.0"
              "GRADIO_SERVER_PORT=7860"
              "DOJO_LLM_BACKEND=llamacpp"
            ];
            
            WorkingDir = "/app";
            
            Cmd = [ "${mkVulkanPythonEnv}/bin/python" "-m" "kataforge.cli.main" "ui" ];
            
            ExposedPorts = {
              "7860/tcp" = {};
            };
            
            Healthcheck = {
              Test = [ "CMD" "${pkgsVulkan.curl}/bin/curl" "-f" "http://localhost:7860/" ];
              Interval = 30000000000;
              Timeout = 10000000000;
              Retries = 3;
              StartPeriod = 60000000000;
            };
            
            Labels = {
              "org.opencontainers.image.title" = "KataForge v.1 UI";
              "org.opencontainers.image.description" = "KataForge v.1 Gradio UI with Vulkan support";
              "org.opencontainers.image.version" = "0.1.0";
              "org.opencontainers.image.vendor" = "DeMoD LLC";
              "com.demod.dojo.component" = "gradio-ui";
            };
          };
        };

        cpuGradioDockerImage = pkgs.dockerTools.buildLayeredImage {
          name = "kataforge-ui";
          tag = "cpu";
          
          contents = [
            (mkDojoGradioApp mkCpuPythonEnv pkgs)
            mkCpuPythonEnv
            pkgs.bash
            pkgs.coreutils
            pkgs.cacert
            pkgs.curl
          ];
          
          config = {
            Env = [
              "PATH=/bin:${mkCpuPythonEnv}/bin"
              "PYTHONPATH=/lib/kataforge"
              "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              "GRADIO_SERVER_NAME=0.0.0.0"
              "GRADIO_SERVER_PORT=7860"
              "DOJO_LLM_BACKEND=ollama"
            ];
            
            WorkingDir = "/app";
            
            Cmd = [ "${mkCpuPythonEnv}/bin/python" "-m" "kataforge.cli.main" "ui" ];
            
            ExposedPorts = {
              "7860/tcp" = {};
            };
            
            Healthcheck = {
              Test = [ "CMD" "${pkgs.curl}/bin/curl" "-f" "http://localhost:7860/" ];
              Interval = 30000000000;
              Timeout = 10000000000;
              Retries = 3;
              StartPeriod = 60000000000;
            };
            
            Labels = {
              "org.opencontainers.image.title" = "KataForge v.1 UI";
              "org.opencontainers.image.description" = "KataForge v.1 Gradio UI (CPU only)";
              "org.opencontainers.image.version" = "0.1.0";
              "org.opencontainers.image.vendor" = "DeMoD LLC";
              "com.demod.dojo.component" = "gradio-ui";
            };
          };
        };

        mkFullDockerImage = { name, tag, pkgSet, pythonEnv, extraContents ? [], extraEnv ? [] }:
          pkgSet.dockerTools.buildLayeredImage {
            inherit name tag;
            
            contents = [
              (mkDojoManagerApp pythonEnv pkgSet)
              (mkDojoGradioApp pythonEnv pkgSet)
              pythonEnv
              pkgSet.bash
              pkgSet.coreutils
              pkgSet.cacert
              pkgSet.curl
              pkgSet.procps
              pkgSet.gnugrep
            ] ++ extraContents;
            
            extraCommands = ''
              mkdir -p app data
              cp ${./docker/entrypoint-full.sh} app/entrypoint.sh
              chmod +x app/entrypoint.sh
            '';
            
            config = {
              Env = [
                "PATH=/bin:/app:${pythonEnv}/bin"
                "PYTHONPATH=/lib/kataforge"
                "SSL_CERT_FILE=${pkgSet.cacert}/etc/ssl/certs/ca-bundle.crt"
                "DOJO_ENVIRONMENT=production"
                "DOJO_LOG_FORMAT=json"
                "DOJO_API_HOST=0.0.0.0"
                "DOJO_API_PORT=8000"
                "GRADIO_SERVER_NAME=0.0.0.0"
                "GRADIO_SERVER_PORT=7860"
              ] ++ extraEnv;
              
              WorkingDir = "/app";
              
              Entrypoint = [ "/app/entrypoint.sh" ];
              
              ExposedPorts = {
                "8000/tcp" = {};
                "7860/tcp" = {};
                "11434/tcp" = {};
              };
              
              Volumes = {
                "/data" = {};
                "/models" = {};
              };
              
              Labels = {
                "org.opencontainers.image.title" = "KataForge v.1 Full";
                "org.opencontainers.image.description" = "KataForge v.1 Full Stack - API + Gradio UI + LLM";
                "org.opencontainers.image.version" = "0.1.0";
                "org.opencontainers.image.vendor" = "DeMoD LLC";
                "com.demod.dojo.type" = "full-stack";
              };
            };
          };

        rocmFullDockerImage = mkFullDockerImage {
          name = "kataforge-full";
          tag = "rocm";
          pkgSet = pkgsRocm;
          pythonEnv = mkRocmPythonEnv;
          extraContents = rocmPackages ++ [ pkgsRocm.ollama ];
          extraEnv = [
            "ROCM_PATH=${pkgsRocm.rocmPackages.clr}"
            "HIP_PATH=${pkgsRocm.rocmPackages.clr}"
            "HSA_OVERRIDE_GFX_VERSION=11.0.0"
            "PYTORCH_ROCM_ARCH=gfx1100"
            "LD_LIBRARY_PATH=${pkgsRocm.rocmPackages.clr}/lib"
            "DOJO_LLM_BACKEND=ollama"
            "DOJO_VISION_MODEL=llava:7b"
            "DOJO_TEXT_MODEL=mistral:7b"
          ];
        };

        cudaFullDockerImage = mkFullDockerImage {
          name = "kataforge-full";
          tag = "cuda";
          pkgSet = pkgsCuda;
          pythonEnv = mkCudaPythonEnv;
          extraContents = cudaDeps ++ [ pkgsCuda.ollama ];
          extraEnv = [
            "CUDA_HOME=${pkgsCuda.cudaPackages.cudatoolkit}"
            "LD_LIBRARY_PATH=${pkgsCuda.cudaPackages.cudatoolkit}/lib"
            "DOJO_LLM_BACKEND=ollama"
            "DOJO_VISION_MODEL=llava:7b"
            "DOJO_TEXT_MODEL=mistral:7b"
          ];
        };

        vulkanFullDockerImage = mkFullDockerImage {
          name = "kataforge-full";
          tag = "vulkan";
          pkgSet = pkgsVulkan;
          pythonEnv = mkVulkanPythonEnv;
          extraContents = (vulkanDeps pkgsVulkan) ++ [ llamaCppVulkan ggufModels ];
          extraEnv = [
            "DOJO_LLM_BACKEND=llamacpp"
            "LLAMA_MODEL_PATH=/models/llava-v1.5-7b-q4_k.gguf"
            "LLAMA_MMPROJ_PATH=/models/mmproj-model-f16.gguf"
          ];
        };

        cpuFullDockerImage = mkFullDockerImage {
          name = "kataforge-full";
          tag = "cpu";
          pkgSet = pkgs;
          pythonEnv = mkCpuPythonEnv;
          extraContents = [ pkgs.ollama ];
          extraEnv = [
            "DOJO_LLM_BACKEND=ollama"
            "DOJO_VISION_MODEL=llava:7b"
            "DOJO_TEXT_MODEL=mistral:7b"
          ];
        };

        llamaCppDockerImage = pkgsVulkan.dockerTools.buildLayeredImage {
          name = "kataforge-llama-cpp";
          tag = "vulkan";
          
          contents = [
            pkgsVulkan.bash
            pkgsVulkan.coreutils
            pkgsVulkan.cacert
            pkgsVulkan.curl
          ] ++ (vulkanDeps pkgsVulkan);
          
          config = {
            Env = [
              "PATH=/bin"
              "LLAMA_ARG_HOST=0.0.0.0"
              "LLAMA_ARG_PORT=8080"
              "LLAMA_ARG_N_GPU_LAYERS=999"
            ];
            
            WorkingDir = "/app";
            
            ExposedPorts = {
              "8080/tcp" = {};
            };
            
            Volumes = {
              "/models" = {};
            };
            
            Labels = {
              "org.opencontainers.image.title" = "KataForge v.1 llama.cpp";
              "org.opencontainers.image.description" = "llama.cpp server with Vulkan backend";
              "org.opencontainers.image.version" = "0.1.0";
              "org.opencontainers.image.vendor" = "DeMoD LLC";
            };
          };
        };

      in {
        packages = {
          default = mkDojoManagerApp mkCpuPythonEnv pkgs;
          cpu = mkDojoManagerApp mkCpuPythonEnv pkgs;
          rocm = mkDojoManagerApp mkRocmPythonEnv pkgsRocm;
          cuda = mkDojoManagerApp mkCudaPythonEnv pkgsCuda;
          vulkan = mkDojoManagerApp mkVulkanPythonEnv pkgsVulkan;
          
          gradio-cpu = mkDojoGradioApp mkCpuPythonEnv pkgs;
          gradio-rocm = mkDojoGradioApp mkRocmPythonEnv pkgsRocm;
          gradio-cuda = mkDojoGradioApp mkCudaPythonEnv pkgsCuda;
          gradio-vulkan = mkDojoGradioApp mkVulkanPythonEnv pkgsVulkan;
          
          python-cpu = mkCpuPythonEnv;
          python-rocm = mkRocmPythonEnv;
          python-cuda = mkCudaPythonEnv;
          python-vulkan = mkVulkanPythonEnv;
          
          docker-cpu = cpuDockerImage;
          docker-rocm = rocmDockerImage;
          docker-cuda = cudaDockerImage;
          docker-vulkan = vulkanDockerImage;
          
          docker-gradio-cpu = cpuGradioDockerImage;
          docker-gradio-rocm = rocmGradioDockerImage;
          docker-gradio-cuda = cudaGradioDockerImage;
          docker-gradio-vulkan = vulkanGradioDockerImage;
          
          docker-full-cpu = cpuFullDockerImage;
          docker-full-rocm = rocmFullDockerImage;
          docker-full-cuda = cudaFullDockerImage;
          docker-full-vulkan = vulkanFullDockerImage;
          
          docker-llama-cpp = llamaCppDockerImage;
        };

        devShells = {
          default = cpuShell;
          cpu = cpuShell;
          rocm = rocmShell;
          cuda = cudaShell;
          vulkan = vulkanShell;
        };

        apps = {
          default = {
            type = "app";
            program = "${mkDojoManagerApp mkCpuPythonEnv pkgs}/bin/kataforge";
            meta = {
              description = "KataForge v.1 CLI - Martial Arts Coach Preservation & Training System";
              mainProgram = "kataforge";
            };
          };
          
          server = {
            type = "app";
            program = "${mkDojoManagerApp mkCpuPythonEnv pkgs}/bin/kataforge-server";
            meta = {
              description = "KataForge v.1 API Server";
              mainProgram = "kataforge-server";
            };
          };
          
          ui = {
            type = "app";
            program = "${mkDojoGradioApp mkCpuPythonEnv pkgs}/bin/kataforge-ui";
            meta = {
              description = "KataForge v.1 Gradio UI";
              mainProgram = "kataforge-ui";
            };
          };
        };

        formatter = pkgs.nixpkgs-fmt;
      }
    );
}