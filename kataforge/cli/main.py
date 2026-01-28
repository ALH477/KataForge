"""
Modern CLI for KataForge using Typer and Rich.

Features:
- Beautiful colored output with Rich
- Progress bars for long operations  
- Connected to real implementations
- Structured command groups
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import typer
    from typer import Argument, Option, Typer
    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False
    typer = None

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.tree import Tree
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None


# =============================================================================
# Console Setup
# =============================================================================

if RICH_AVAILABLE:
    console = Console()
    err_console = Console(stderr=True)
else:
    console = None
    err_console = None


def print_error(message: str) -> None:
    """Print error message."""
    if RICH_AVAILABLE:
        err_console.print(f"[bold red]Error:[/bold red] {message}")
    else:
        print(f"Error: {message}", file=sys.stderr)


def print_success(message: str) -> None:
    """Print success message."""
    if RICH_AVAILABLE:
        console.print(f"[bold green]✓[/bold green] {message}")
    else:
        print(f"✓ {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]⚠[/bold yellow] {message}")
    else:
        print(f"⚠ {message}")


def print_info(message: str) -> None:
    """Print info message."""
    if RICH_AVAILABLE:
        console.print(f"[bold blue]ℹ[/bold blue] {message}")
    else:
        print(f"ℹ {message}")


# =============================================================================
# App Setup
# =============================================================================

if TYPER_AVAILABLE:
    app = Typer(
        name="kataforge",
        help="Train smarter with AI-powered technique analysis",
        add_completion=True,
        rich_markup_mode="rich",
        no_args_is_help=True,
    )
    
    # Subcommand groups
    coach_app = Typer(help="Manage coach profiles")
    app.add_typer(coach_app, name="coach")
else:
    app = None
    coach_app = None


# =============================================================================
# Helper: Get Data Directory
# =============================================================================

def get_data_dir() -> Path:
    """Get the data directory from settings or default."""
    try:
        from ..core.settings import get_settings
        settings = get_settings()
        return Path(settings.data_dir).expanduser()
    except Exception:
        return Path("~/.dojo/data").expanduser()


def get_profiles_dir() -> Path:
    """Get the profiles directory."""
    return get_data_dir() / "profiles"


# =============================================================================
# Init Command
# =============================================================================

if TYPER_AVAILABLE:
    @app.command()
    def init(
        name: str = Option("dojo", "--name", "-n", help="Your training space name"),
        data_dir: str = Option("~/.dojo/data", "--data-dir", "-d", help="Where to store your training data"),
    ) -> None:
        """Set up your KataForge training space."""
        if RICH_AVAILABLE:
            console.print(Panel(
                f"[bold]Setting Up KataForge[/bold]\n"
                f"Name: [cyan]{name}[/cyan]\n"
                f"Data: [cyan]{data_dir}[/cyan]",
                title="Training Space",
                border_style="blue",
            ))
        else:
            print(f"Initializing dojo system '{name}' with data directory '{data_dir}'")
        
        base_dir = Path(data_dir).expanduser()
        
        directories = [
            ("raw", "Raw video footage"),
            ("processed", "Preprocessed videos"),
            ("poses", "Extracted pose data"),
            ("models", "Trained models"),
            ("profiles", "Coach profiles"),
            ("exports", "Exported analyses"),
            ("checkpoints", "Training checkpoints"),
            ("logs", "Training logs"),
        ]
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Creating directories...", total=len(directories))
                
                for dirname, desc in directories:
                    dir_path = base_dir / dirname
                    dir_path.mkdir(parents=True, exist_ok=True)
                    progress.update(task, advance=1, description=f"Creating {dirname}...")
        else:
            for dirname, desc in directories:
                dir_path = base_dir / dirname
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  Created {dirname}/")
        
        # Show result
        if RICH_AVAILABLE:
            tree = Tree(f"[bold]{base_dir}[/bold]")
            for dirname, desc in directories:
                tree.add(f"[cyan]{dirname}/[/cyan] [dim]{desc}[/dim]")
            console.print(tree)
        
        print_success("Training space is ready! 🎯")


# =============================================================================
# Extract Pose Command (Real Implementation)
# =============================================================================

if TYPER_AVAILABLE:
    @app.command("extract-pose")
    def extract_pose(
        video: str = Argument(..., help="Video file to analyze"),
        output: str = Option(..., "--output", "-o", help="Save pose data here"),
        model_complexity: int = Option(2, "--model", "-m", help="Detection accuracy: 0 (fast), 1 (balanced), 2 (precise)"),
        confidence: float = Option(0.7, "--confidence", "-c", help="Minimum confidence for movement detection (0.0-1.0)"),
    ) -> None:
        """Extract movement patterns from video for AI analysis."""
        video_path = Path(video)
        output_path = Path(output)
        
        if not video_path.exists():
            print_error(f"Video file not found: {video}")
            raise typer.Exit(code=1)
        
        if RICH_AVAILABLE:
            console.print(Panel(
                f"[bold]Movement Analysis[/bold]\n"
                f"Video: [cyan]{video}[/cyan]\n"
                f"Output: [cyan]{output}[/cyan]\n"
                f"Accuracy: [yellow]{model_complexity}[/yellow]\n"
                f"Confidence: [yellow]{confidence}[/yellow]",
                title="Extract Movement",
                border_style="blue",
            ))
        else:
            print(f"Extracting poses from {video} → {output}")
        
        # Try to use real implementation
        try:
            from ..preprocessing.mediapipe_wrapper import MediaPipePoseExtractor
            
            extractor = MediaPipePoseExtractor(
                model_complexity=model_complexity,
                min_detection_confidence=confidence,
                min_tracking_confidence=confidence,
            )
            
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task("Extracting poses from video...", total=None)
                    pose_data = extractor.extract_from_video(str(video_path))
                    progress.update(task, completed=True)
            else:
                print("  Extracting poses...")
                pose_data = extractor.extract_from_video(str(video_path))
            
            # Save the poses
            extractor.save_poses(pose_data, str(output_path))
            
            # Show summary
            if RICH_AVAILABLE:
                table = Table(title="Extraction Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                table.add_row("Total Frames", str(pose_data['total_frames']))
                table.add_row("FPS", f"{pose_data['fps']:.2f}")
                table.add_row("Duration", f"{pose_data['total_frames'] / pose_data['fps']:.2f}s")
                table.add_row("Landmarks", "33 (MediaPipe)")
                console.print(table)
            
            print_success(f"Movement patterns captured to {output} 🎯")
            
        except ImportError as e:
            print_error(f"MediaPipe/OpenCV not available: {e}")
            print_info("Install with: pip install opencv-python mediapipe")
            raise typer.Exit(code=1)
        except Exception as e:
            print_error(f"Extraction failed: {e}")
            raise typer.Exit(code=1)


# =============================================================================
# Train Command (Real Implementation)
# =============================================================================

if TYPER_AVAILABLE:
    @app.command()
    def train(
        coach_id: str = Argument(..., help="Training coach ID"),
        data_dir: str = Option(None, "--data-dir", "-d", help="Training data location"),
        epochs: int = Option(100, "--epochs", "-e", help="Number of training cycles"),
        batch_size: int = Option(16, "--batch-size", "-b", help="Videos processed together"),
        learning_rate: float = Option(0.001, "--lr", help="Learning speed (smaller = slower, more careful)"),
        device: str = Option("auto", "--device", help="Processing power: cpu, cuda, rocm, or auto"),
        checkpoint_dir: str = Option(None, "--checkpoint-dir", help="Save progress here"),
        resume: Optional[str] = Option(None, "--resume", "-r", help="Continue from this checkpoint"),
    ) -> None:
        """Train an AI model to analyze techniques like this coach."""
        # Resolve directories
        base_dir = get_data_dir()
        training_data_dir = Path(data_dir) if data_dir else base_dir / "poses" / coach_id
        checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else base_dir / "checkpoints" / coach_id
        
        if RICH_AVAILABLE:
            console.print(Panel(
                f"[bold]Model Training[/bold]\n"
                f"Coach: [cyan]{coach_id}[/cyan]\n"
                f"Data: [cyan]{training_data_dir}[/cyan]\n"
                f"Epochs: [yellow]{epochs}[/yellow]\n"
                f"Batch Size: [yellow]{batch_size}[/yellow]\n"
                f"Learning Rate: [yellow]{learning_rate}[/yellow]\n"
                f"Device: [yellow]{device}[/yellow]",
                title="Train",
                border_style="blue",
            ))
        else:
            print(f"Training model for coach '{coach_id}'")
        
        # Check if data directory exists
        if not training_data_dir.exists():
            print_error(f"Training data directory not found: {training_data_dir}")
            print_info(f"Extract poses first with: kataforge extract-pose <video> -o {training_data_dir}/<technique>/<video>.json")
            raise typer.Exit(code=1)
        
        try:
            from ..ml.data_loader import create_data_loaders
            from ..ml.models import FormAssessor
            from ..ml.trainer import Trainer
            import torch
            
            # Resolve device
            if device == "auto":
                actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                actual_device = device
            
            print_info(f"Using device: {actual_device}")
            
            # Create data loaders
            if RICH_AVAILABLE:
                with console.status("Loading training data..."):
                    train_loader, val_loader = create_data_loaders(
                        str(training_data_dir),
                        batch_size=batch_size,
                        coaches=[coach_id],
                    )
            else:
                print("  Loading training data...")
                train_loader, val_loader = create_data_loaders(
                    str(training_data_dir),
                    batch_size=batch_size,
                    coaches=[coach_id],
                )
            
            # Create model
            model = FormAssessor()
            
            # Create trainer
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                learning_rate=learning_rate,
                device=actual_device,
            )
            
            # Load checkpoint if resuming
            if resume:
                trainer.load_model(resume)
                print_info(f"Resumed from checkpoint: {resume}")
            
            # Train with progress
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Training...", total=epochs)
                    
                    # Custom training loop with progress updates
                    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
                    best_acc = 0.0
                    
                    for epoch in range(epochs):
                        train_loss = trainer.train_epoch()
                        val_metrics = trainer.validate()
                        
                        history['train_loss'].append(train_loss)
                        history['val_loss'].append(val_metrics['loss'])
                        history['val_accuracy'].append(val_metrics['accuracy'])
                        
                        progress.update(
                            task, 
                            advance=1,
                            description=f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {val_metrics['accuracy']:.1f}%"
                        )
                        
                        # Save best checkpoint
                        if val_metrics['accuracy'] > best_acc:
                            best_acc = val_metrics['accuracy']
                            checkpoint_path.mkdir(parents=True, exist_ok=True)
                            trainer.save_model(str(checkpoint_path / f"best_model.pt"))
            else:
                history = trainer.train(
                    epochs=epochs,
                    save_checkpoint=True,
                    checkpoint_dir=str(checkpoint_path),
                )
            
            # Show results
            if RICH_AVAILABLE:
                table = Table(title="Training Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                table.add_row("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
                table.add_row("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
                table.add_row("Best Accuracy", f"{max(history['val_accuracy']):.2f}%")
                table.add_row("Checkpoint", str(checkpoint_path / "best_model.pt"))
                console.print(table)
            
            # Save training history
            history_path = checkpoint_path / "training_history.json"
            trainer.save_history(history, str(history_path))
            
            print_success("Training complete! Model is ready to analyze techniques 🎉")
            
        except ImportError as e:
            print_error(f"PyTorch not available: {e}")
            print_info("Install with GPU support via nix develop .#cuda or .#rocm")
            raise typer.Exit(code=1)
        except Exception as e:
            print_error(f"Training failed: {e}")
            raise typer.Exit(code=1)


# =============================================================================
# Analyze Command (Real Implementation)
# =============================================================================

if TYPER_AVAILABLE:
    @app.command()
    def analyze(
        video: str = Argument(..., help="Video file to analyze"),
        coach: str = Option(..., "--coach", "-c", help="Which coach's style to match"),
        technique: str = Option(..., "--technique", "-t", help="Name of the technique"),
        model_path: Optional[str] = Option(None, "--model", "-m", help="Path to trained model"),
        output: Optional[str] = Option(None, "--output", "-o", help="Save analysis to file"),
        verbose: bool = Option(False, "--verbose", "-v", help="Show detailed breakdown"),
    ) -> None:
        """Analyze your technique and get detailed feedback."""
        video_path = Path(video)
        
        if not video_path.exists():
            print_error(f"Video file not found: {video}")
            raise typer.Exit(code=1)
        
        if RICH_AVAILABLE:
            console.print(Panel(
                f"[bold]Technique Analysis[/bold]\n"
                f"Video: [cyan]{video}[/cyan]\n"
                f"Coach: [cyan]{coach}[/cyan]\n"
                f"Technique: [cyan]{technique}[/cyan]",
                title="Analyze",
                border_style="gold",
            ))
        else:
            print(f"Analyzing technique '{technique}' using coach '{coach}'")
        
        try:
            from ..preprocessing.mediapipe_wrapper import MediaPipePoseExtractor
            from ..biomechanics.calculator import BiomechanicsCalculator
            
            # Extract poses
            if RICH_AVAILABLE:
                with console.status("Extracting poses from video..."):
                    extractor = MediaPipePoseExtractor()
                    pose_data = extractor.extract_from_video(str(video_path))
            else:
                print("  Extracting poses...")
                extractor = MediaPipePoseExtractor()
                pose_data = extractor.extract_from_video(str(video_path))
            
            # Calculate biomechanics
            if RICH_AVAILABLE:
                with console.status("Calculating biomechanics..."):
                    calculator = BiomechanicsCalculator()
                    biomechanics = calculator.calculate_all(pose_data['poses'], pose_data['fps'])
            else:
                print("  Calculating biomechanics...")
                calculator = BiomechanicsCalculator()
                biomechanics = calculator.calculate_all(pose_data['poses'], pose_data['fps'])
            
            # Try to load model for scoring
            analysis_result = {
                'video': str(video_path),
                'coach': coach,
                'technique': technique,
                'frames': pose_data['total_frames'],
                'fps': pose_data['fps'],
                'biomechanics': biomechanics,
            }
            
            # Try ML-based scoring
            try:
                import torch
                from ..ml.models import FormAssessor
                
                # Load model
                resolved_model_path = model_path or (get_data_dir() / "checkpoints" / coach / "best_model.pt")
                
                if Path(resolved_model_path).exists():
                    model = FormAssessor()
                    model.load_state_dict(torch.load(resolved_model_path, weights_only=True))
                    model.eval()
                    
                    # Prepare data
                    poses_tensor = torch.FloatTensor(pose_data['poses']).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(poses_tensor)
                    
                    analysis_result['overall_score'] = float(output['overall_score'].item())
                    analysis_result['aspect_scores'] = {
                        'speed': float(output['aspect_scores'][0, 0].item()),
                        'force': float(output['aspect_scores'][0, 1].item()),
                        'timing': float(output['aspect_scores'][0, 2].item()),
                        'balance': float(output['aspect_scores'][0, 3].item()),
                        'coordination': float(output['aspect_scores'][0, 4].item()),
                    }
                else:
                    print_warning(f"No trained model found at {resolved_model_path}")
                    # Use biomechanics-based heuristic scoring
                    analysis_result['overall_score'] = _heuristic_score(biomechanics)
                    analysis_result['aspect_scores'] = _heuristic_aspects(biomechanics)
                    
            except ImportError:
                # Use heuristic scoring without ML
                analysis_result['overall_score'] = _heuristic_score(biomechanics)
                analysis_result['aspect_scores'] = _heuristic_aspects(biomechanics)
            
            # Generate corrections and recommendations
            analysis_result['corrections'] = _generate_corrections(analysis_result)
            analysis_result['recommendations'] = _generate_recommendations(analysis_result)
            
            # Display results
            _display_analysis(analysis_result, verbose)
            
            # Save to file if requested
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(analysis_result, f, indent=2, default=str)
                print_success(f"Analysis saved to {output}")
                
        except ImportError as e:
            print_error(f"Required dependencies not available: {e}")
            print_info("Install with: pip install opencv-python mediapipe")
            raise typer.Exit(code=1)
        except Exception as e:
            print_error(f"Analysis failed: {e}")
            raise typer.Exit(code=1)


def _heuristic_score(biomechanics: Dict[str, Any]) -> float:
    """Generate overall score from biomechanics."""
    # Simple heuristic based on biomechanics values
    score = 7.0  # Base score
    
    # Adjust based on available metrics
    if 'max_speed' in biomechanics:
        speed = biomechanics['max_speed']
        if speed > 5.0:
            score += 0.5
        elif speed < 2.0:
            score -= 0.5
    
    if 'kinetic_chain_efficiency' in biomechanics:
        efficiency = biomechanics['kinetic_chain_efficiency']
        score += (efficiency - 80) / 20  # Adjust by efficiency
    
    return max(1.0, min(10.0, score))


def _heuristic_aspects(biomechanics: Dict[str, Any]) -> Dict[str, float]:
    """Generate aspect scores from biomechanics."""
    return {
        'speed': min(10.0, 5.0 + biomechanics.get('max_speed', 3.0)),
        'force': min(10.0, 5.0 + biomechanics.get('peak_force', 1000) / 500),
        'timing': 7.5,  # Default without ML
        'balance': 8.0,  # Default without ML
        'coordination': biomechanics.get('kinetic_chain_efficiency', 75) / 10,
    }


def _generate_corrections(result: Dict[str, Any]) -> List[str]:
    """Generate corrections based on analysis."""
    corrections = []
    aspects = result.get('aspect_scores', {})
    
    if aspects.get('timing', 10) < 7.0:
        corrections.append("Improve hip rotation timing - initiate rotation earlier")
    if aspects.get('balance', 10) < 7.0:
        corrections.append("Maintain better center of gravity throughout the technique")
    if aspects.get('coordination', 10) < 7.0:
        corrections.append("Synchronize arm and leg movements more precisely")
    if aspects.get('speed', 10) < 6.0:
        corrections.append("Increase technique speed - focus on explosive power")
    if aspects.get('force', 10) < 6.0:
        corrections.append("Generate more force through proper weight transfer")
    
    return corrections if corrections else ["Good form overall - continue practicing"]


def _generate_recommendations(result: Dict[str, Any]) -> List[str]:
    """Generate training recommendations."""
    recommendations = [
        "Practice shadowboxing for 10 minutes daily",
        "Focus on chambering technique before strikes",
        "Drill combination flows for muscle memory",
    ]
    
    aspects = result.get('aspect_scores', {})
    
    if aspects.get('speed', 10) < 7.0:
        recommendations.append("Add plyometric exercises to improve explosive speed")
    if aspects.get('balance', 10) < 7.0:
        recommendations.append("Practice single-leg balance exercises")
    
    return recommendations


def _display_analysis(result: Dict[str, Any], verbose: bool) -> None:
    """Display analysis results."""
    if RICH_AVAILABLE:
        console.print()
        
        # Overall score
        score = result.get('overall_score', 0)
        score_color = "green" if score >= 8.0 else "yellow" if score >= 6.0 else "red"
        console.print(Panel(
            f"[bold {score_color}]{score:.1f}[/bold {score_color}] / 10",
            title="Overall Score",
            border_style=score_color,
        ))
        
        # Aspect scores
        aspects = result.get('aspect_scores', {})
        if aspects:
            table = Table(title="Aspect Scores")
            table.add_column("Aspect", style="cyan")
            table.add_column("Score", justify="right")
            table.add_column("Rating", justify="center")
            
            for aspect, score in aspects.items():
                color = "green" if score >= 8.0 else "yellow" if score >= 6.0 else "red"
                rating = "Excellent" if score >= 8.5 else "Very Good" if score >= 7.5 else "Good" if score >= 6.5 else "Needs Work"
                table.add_row(aspect.title(), f"[{color}]{score:.1f}[/{color}]", rating)
            
            console.print(table)
        
        # Biomechanics (verbose mode)
        if verbose and 'biomechanics' in result:
            bio = result['biomechanics']
            bio_table = Table(title="Biomechanics")
            bio_table.add_column("Metric", style="cyan")
            bio_table.add_column("Value", justify="right")
            
            for key, value in bio.items():
                if isinstance(value, float):
                    bio_table.add_row(key.replace('_', ' ').title(), f"{value:.2f}")
                else:
                    bio_table.add_row(key.replace('_', ' ').title(), str(value))
            
            console.print(bio_table)
        
        # Corrections
        corrections = result.get('corrections', [])
        if corrections:
            console.print()
            console.print("[bold]Corrections:[/bold]")
            for correction in corrections:
                console.print(f"  [yellow]•[/yellow] {correction}")
        
        # Recommendations
        recommendations = result.get('recommendations', [])
        if recommendations:
            console.print()
            console.print("[bold]Recommendations:[/bold]")
            for rec in recommendations:
                console.print(f"  [blue]•[/blue] {rec}")
    else:
        print(f"\nOverall Score: {result.get('overall_score', 0):.1f}/10")
        for aspect, score in result.get('aspect_scores', {}).items():
            print(f"  {aspect}: {score:.1f}")


# =============================================================================
# Coach Commands (Real Implementation)
# =============================================================================

if TYPER_AVAILABLE:
    @coach_app.command("add")
    def coach_add(
        coach_id: str = Argument(..., help="Unique coach identifier"),
        name: str = Option(..., "--name", "-n", help="Coach's full name"),
        style: str = Option(..., "--style", "-s", help="Martial art style"),
        rank: str = Option(..., "--rank", "-r", help="Rank or belt level"),
        years_experience: int = Option(0, "--years", "-y", help="Years of experience"),
    ) -> None:
        """Add a new coach profile to the system."""
        try:
            from ..profiles.manager import ProfileManager
            
            profiles_dir = get_profiles_dir()
            manager = ProfileManager(str(profiles_dir))
            
            profile = manager.create_profile(
                coach_id=coach_id,
                name=name,
                style=style,
                rank=rank,
                years_experience=years_experience,
            )
            
            if RICH_AVAILABLE:
                console.print(Panel(
                    f"[bold]Coach Profile Created[/bold]\n"
                    f"ID: [cyan]{coach_id}[/cyan]\n"
                    f"Name: [cyan]{name}[/cyan]\n"
                    f"Style: [yellow]{style}[/yellow]\n"
                    f"Rank: [yellow]{rank}[/yellow]",
                    title="Success",
                    border_style="green",
                ))
            else:
                print(f"Created profile for {name} ({coach_id})")
            
            print_success(f"Coach profile '{coach_id}' created")
            
        except Exception as e:
            print_error(f"Failed to create profile: {e}")
            raise typer.Exit(code=1)


    @coach_app.command("list")
    def coach_list(
        style: Optional[str] = Option(None, "--style", "-s", help="Filter by martial art style"),
    ) -> None:
        """List all registered coach profiles."""
        try:
            from ..profiles.manager import ProfileManager
            
            profiles_dir = get_profiles_dir()
            manager = ProfileManager(str(profiles_dir))
            
            coach_ids = manager.list_profiles()
            
            if not coach_ids:
                print_warning("No coach profiles found")
                print_info(f"Add a coach with: kataforge coach add <id> --name <name> --style <style> --rank <rank>")
                return
            
            coaches = []
            for coach_id in coach_ids:
                profile = manager.load_profile(coach_id)
                if profile:
                    if style and style.lower() not in profile.get('style', '').lower():
                        continue
                    coaches.append(profile)
            
            if RICH_AVAILABLE:
                table = Table(title="Registered Coaches")
                table.add_column("ID", style="cyan")
                table.add_column("Name", style="white")
                table.add_column("Style", style="yellow")
                table.add_column("Rank", style="green")
                table.add_column("Experience", style="dim")
                
                for coach in coaches:
                    table.add_row(
                        coach.get('id', ''),
                        coach.get('name', ''),
                        coach.get('style', ''),
                        coach.get('rank', ''),
                        f"{coach.get('years_experience', 0)} years",
                    )
                
                console.print(table)
            else:
                print("Coaches:")
                for coach in coaches:
                    print(f"  - {coach.get('id')}: {coach.get('name')} ({coach.get('style')}, {coach.get('rank')})")
                    
        except Exception as e:
            print_error(f"Failed to list profiles: {e}")
            raise typer.Exit(code=1)


    @coach_app.command("show")
    def coach_show(
        coach_id: str = Argument(..., help="Coach identifier"),
    ) -> None:
        """Show detailed information about a coach."""
        try:
            from ..profiles.manager import ProfileManager
            
            profiles_dir = get_profiles_dir()
            manager = ProfileManager(str(profiles_dir))
            
            profile = manager.load_profile(coach_id)
            
            if not profile:
                print_error(f"Coach profile not found: {coach_id}")
                raise typer.Exit(code=1)
            
            if RICH_AVAILABLE:
                # Build profile display
                techniques = profile.get('techniques', [])
                techniques_str = ', '.join(techniques) if techniques else 'None recorded'
                
                console.print(Panel(
                    f"[bold]{profile.get('name', 'Unknown')}[/bold]\n\n"
                    f"[cyan]ID:[/cyan] {profile.get('id', '')}\n"
                    f"[cyan]Style:[/cyan] {profile.get('style', '')}\n"
                    f"[cyan]Rank:[/cyan] {profile.get('rank', '')}\n"
                    f"[cyan]Experience:[/cyan] {profile.get('years_experience', 0)} years\n"
                    f"[cyan]Created:[/cyan] {profile.get('created_at', 'Unknown')[:10]}\n\n"
                    f"[bold]Techniques:[/bold]\n  {techniques_str}",
                    title=f"Coach Profile: {coach_id}",
                    border_style="blue",
                ))
            else:
                print(f"Coach: {coach_id}")
                print(f"  Name: {profile.get('name')}")
                print(f"  Style: {profile.get('style')}")
                print(f"  Rank: {profile.get('rank')}")
                
        except Exception as e:
            print_error(f"Failed to show profile: {e}")
            raise typer.Exit(code=1)


    @coach_app.command("delete")
    def coach_delete(
        coach_id: str = Argument(..., help="Coach identifier"),
        force: bool = Option(False, "--force", "-f", help="Skip confirmation"),
    ) -> None:
        """Delete a coach profile."""
        try:
            from ..profiles.manager import ProfileManager
            
            profiles_dir = get_profiles_dir()
            manager = ProfileManager(str(profiles_dir))
            
            profile = manager.load_profile(coach_id)
            if not profile:
                print_error(f"Coach profile not found: {coach_id}")
                raise typer.Exit(code=1)
            
            if not force:
                confirm = typer.confirm(f"Are you sure you want to delete coach '{coach_id}'?")
                if not confirm:
                    print_info("Cancelled")
                    return
            
            if manager.delete_profile(coach_id):
                print_success(f"Coach profile '{coach_id}' deleted")
            else:
                print_error(f"Failed to delete profile")
                
        except Exception as e:
            print_error(f"Failed to delete profile: {e}")
            raise typer.Exit(code=1)


# =============================================================================
# Serve Command
# =============================================================================

if TYPER_AVAILABLE:
    @app.command()
    def serve(
        host: str = Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
        port: int = Option(8000, "--port", "-p", help="Port to bind to"),
        workers: int = Option(1, "--workers", "-w", help="Number of worker processes"),
        reload: bool = Option(False, "--reload", "-r", help="Enable auto-reload (development)"),
    ) -> None:
        """Start the API server."""
        if RICH_AVAILABLE:
            console.print(Panel(
                f"[bold]Starting API Server[/bold]\n"
                f"Host: [cyan]{host}[/cyan]\n"
                f"Port: [cyan]{port}[/cyan]\n"
                f"Workers: [yellow]{workers}[/yellow]\n"
                f"Reload: [yellow]{reload}[/yellow]",
                title="Serve",
                border_style="blue",
            ))
        else:
            print(f"Starting server on {host}:{port}")
        
        try:
            import os
            os.environ.setdefault("DOJO_API_HOST", host)
            os.environ.setdefault("DOJO_API_PORT", str(port))
            os.environ.setdefault("DOJO_API_WORKERS", str(workers))
            os.environ.setdefault("DOJO_API_RELOAD", str(reload).lower())
            
            from ..api.server import run_server
            run_server()
        except ImportError as e:
            print_error(f"Failed to import server: {e}")
            print_info("Make sure FastAPI and uvicorn are installed")
            raise typer.Exit(code=1)


# =============================================================================
# UI Command
# =============================================================================

if TYPER_AVAILABLE:
    @app.command()
    def ui(
        host: str = Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
        port: int = Option(7860, "--port", "-p", help="Port to bind to"),
        api_url: str = Option("http://localhost:8000", "--api-url", "-a", help="Dojo API URL"),
        ollama_url: str = Option("http://localhost:11434", "--ollama-url", "-o", help="Ollama/llama.cpp URL"),
        llm_backend: str = Option("ollama", "--llm-backend", "-l", help="LLM backend: ollama or llamacpp"),
        share: bool = Option(False, "--share", "-s", help="Create public Gradio link"),
    ) -> None:
        """Start the Gradio web interface."""
        if RICH_AVAILABLE:
            console.print(Panel(
                f"[bold]Starting Gradio UI[/bold]\n"
                f"Host: [cyan]{host}[/cyan]\n"
                f"Port: [cyan]{port}[/cyan]\n"
                f"API URL: [cyan]{api_url}[/cyan]\n"
                f"LLM URL: [cyan]{ollama_url}[/cyan]\n"
                f"LLM Backend: [yellow]{llm_backend}[/yellow]\n"
                f"Share: [yellow]{share}[/yellow]",
                title="UI",
                border_style="blue",
            ))
        else:
            print(f"Starting Gradio UI on {host}:{port}")
        
        try:
            from ..ui.gradio_app import launch_ui
            launch_ui(
                host=host,
                port=port,
                api_url=api_url,
                ollama_url=ollama_url,
                llm_backend=llm_backend,
                share=share,
            )
        except ImportError as e:
            print_error(f"Failed to import UI: {e}")
            print_info("Make sure Gradio is installed: pip install gradio")
            raise typer.Exit(code=1)


# =============================================================================
# Status Command
# =============================================================================

if TYPER_AVAILABLE:
    @app.command()
    def status() -> None:
        """Show system status and configuration."""
        deps = []
        
        # Python
        deps.append(("Python", sys.version.split()[0], True))
        
        # Check optional deps
        try:
            import torch
            gpu_info = f"CUDA: {torch.cuda.is_available()}"
            if torch.cuda.is_available():
                gpu_info += f", GPUs: {torch.cuda.device_count()}"
            deps.append(("PyTorch", f"{torch.__version__} ({gpu_info})", True))
        except ImportError:
            deps.append(("PyTorch", "Not installed", False))
        
        try:
            import cv2
            deps.append(("OpenCV", cv2.__version__, True))
        except ImportError:
            deps.append(("OpenCV", "Not installed", False))
        
        try:
            import mediapipe
            deps.append(("MediaPipe", mediapipe.__version__, True))
        except ImportError:
            deps.append(("MediaPipe", "Not installed", False))
        
        try:
            import fastapi
            deps.append(("FastAPI", fastapi.__version__, True))
        except ImportError:
            deps.append(("FastAPI", "Not installed", False))
        
        if RICH_AVAILABLE:
            table = Table(title="KataForge Status")
            table.add_column("Component", style="cyan")
            table.add_column("Version", style="white")
            table.add_column("Status", justify="center")
            
            table.add_row("KataForge", "0.1.0", "[green]●[/green]")
            
            for name, version, ok in deps:
                status_icon = "[green]●[/green]" if ok else "[red]●[/red]"
                table.add_row(name, version, status_icon)
            
            console.print(table)
            
            # Configuration
            try:
                from ..core.settings import get_settings
                settings = get_settings()
                
                config_table = Table(title="Configuration")
                config_table.add_column("Setting", style="cyan")
                config_table.add_column("Value", style="white")
                
                env_val = settings.environment.value if hasattr(settings.environment, 'value') else str(settings.environment)
                config_table.add_row("Environment", env_val)
                config_table.add_row("Debug", str(settings.debug))
                config_table.add_row("API Host", settings.api_host)
                config_table.add_row("API Port", str(settings.api_port))
                log_val = settings.log_level.value if hasattr(settings.log_level, 'value') else str(settings.log_level)
                config_table.add_row("Log Level", log_val)
                config_table.add_row("Data Dir", str(get_data_dir()))
                
                console.print(config_table)
            except Exception:
                pass
            
            # Show coach count
            try:
                from ..profiles.manager import ProfileManager
                manager = ProfileManager(str(get_profiles_dir()))
                coach_count = len(manager.list_profiles())
                console.print(f"\n[dim]Registered coaches: {coach_count}[/dim]")
            except Exception:
                pass
        else:
            print("KataForge Status:")
            print("  Version: 0.1.0")
            for name, version, ok in deps:
                status = "OK" if ok else "Missing"
                print(f"  {name}: {version} [{status}]")


# =============================================================================
# Version Callback
# =============================================================================

if TYPER_AVAILABLE:
    def version_callback(value: bool) -> None:
        """Show version and exit."""
        if value:
            if RICH_AVAILABLE:
                console.print("[bold]kataforge[/bold] version [cyan]0.1.0[/cyan]")
            else:
                print("kataforge version 0.1.0")
            raise typer.Exit()


    @app.callback()
    def main_callback(
        version: bool = Option(None, "--version", "-V", callback=version_callback, is_eager=True, help="Show version and exit"),
    ) -> None:
        """KataForge - Train smarter with AI-powered technique analysis."""
        pass


# =============================================================================
# Entry Point
# =============================================================================

def main() -> None:
    """Main entry point for the CLI."""
    if not TYPER_AVAILABLE:
        print("Error: Typer is required for the CLI. Install with: pip install typer", file=sys.stderr)
        sys.exit(1)
    
    app()


if __name__ == "__main__":
    main()
