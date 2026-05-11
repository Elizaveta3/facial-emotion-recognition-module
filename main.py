import tkinter as tk
from tkinter import messagebox

from auth_db import (
    InvalidCredentialsError,
    UserAlreadyExistsError,
    authenticate_user,
    create_user,
    get_user,
    init_db,
    update_face_coordinates,
)
from recognition import (
    CameraUnavailableError,
    FaceNotFoundError,
    capture_face_profile,
    parse_face_profile,
    run_emotion_recognition,
    serialize_face_profile,
)


class EmotionRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        init_db()
        self.title("Emotion Recognition")
        self.geometry("1180x760")
        self.minsize(900, 620)
        self.configure(bg="#eef3f8")
        self.current_user = None
        self.current_frame = None
        self._configure_theme()
        self.attributes("-fullscreen", True)
        self.bind("<F11>", self.toggle_fullscreen)
        self.bind("<Control-q>", lambda event: self.shutdown())
        self.show_home()
        self.after(150, self._activate_window)

    def _configure_theme(self):
        self.colors = {
            "bg": "#eef3f8",
            "panel": "#ffffff",
            "surface": "#f8fafc",
            "text": "#142033",
            "muted": "#667085",
            "primary": "#0f766e",
            "primary_hover": "#115e59",
            "secondary": "#e7edf4",
            "secondary_hover": "#d6e0ea",
            "danger": "#b42318",
            "border": "#b8c4d3",
            "field": "#ffffff",
            "field_focus": "#0f766e",
            "accent": "#164e63",
        }

    def toggle_fullscreen(self, event=None):
        self.attributes("-fullscreen", not self.attributes("-fullscreen"))

    def shutdown(self):
        self.destroy()

    def _activate_window(self):
        self.lift()
        if self.focus_displayof() is None:
            self.focus_force()

    def _set_screen(self, builder):
        if self.current_frame is not None:
            self.current_frame.destroy()
        frame = tk.Frame(self, bg=self.colors["bg"])
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        self.current_frame = frame
        self._topbar(frame)
        builder(frame)
        self.after_idle(self._activate_window)

    def _topbar(self, parent):
        bar = tk.Frame(parent, bg=self.colors["accent"], height=72)
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)

        title = tk.Label(
            bar,
            text="Facial Expression Analysis and Emotion Classification Module",
            bg=self.colors["accent"],
            fg="#ffffff",
            font=("Arial", 16, "bold"),
        )
        title.grid(row=0, column=0, sticky="w", padx=28, pady=20)

        controls = tk.Frame(bar, bg=self.colors["accent"])
        controls.grid(row=0, column=1, sticky="e", padx=18, pady=14)
        self._button(
            controls,
            text="Exit",
            command=self.shutdown,
            danger=True,
            outline=True,
            compact=True,
        ).grid(row=0, column=0)

    def _content_area(self, parent):
        area = tk.Frame(parent, bg=self.colors["bg"])
        area.grid(row=1, column=0, sticky="nsew", padx=34, pady=30)
        area.columnconfigure(0, weight=1)
        area.columnconfigure(1, weight=0)
        area.columnconfigure(2, weight=1)
        area.rowconfigure(0, weight=1)
        return area

    def _center_panel(self, parent, max_width=760):
        area = self._content_area(parent)
        panel_width = min(max_width, max(560, self.winfo_screenwidth() - 120))
        panel = tk.Frame(
            area,
            bg=self.colors["panel"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        panel.grid(row=0, column=1, sticky="nsew")
        area.grid_columnconfigure(1, weight=0, minsize=panel_width)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)
        return panel

    def _label(self, parent, text, size=13, weight="normal", color=None, justify="center"):
        return tk.Label(
            parent,
            text=text,
            bg=parent["bg"],
            fg=color or self.colors["text"],
            font=("Arial", size, weight),
            wraplength=680,
            justify=justify,
        )


    def _button(self, parent, text, command, primary=True, danger=False, outline=False, text_color=None, compact=False):
        if danger:
            fg = text_color or self.colors["danger"]
            bg = parent["bg"] if outline else self.colors["danger"]
            border = self.colors["danger"]
            relief = "solid" if outline else "flat"
            bd = 2 if outline else 0
        else:
            bg = self.colors["primary"] if primary else self.colors["secondary"]
            fg = text_color or ("#ffffff" if primary else self.colors["text"])
            border = self.colors["primary"] if outline else bg
            relief = "solid" if outline else "flat"
            bd = 2 if outline else 0

            if outline:
                bg = parent["bg"]

        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            relief=relief,
            bd=bd,
            highlightthickness=0,
            highlightbackground=border,
            padx=16 if compact else 28,
            pady=8 if compact else 14,
            font=("Arial", 11 if compact else 13, "bold"),
            cursor="hand2",
        )

    def _entry(self, parent, show=None):
        entry = tk.Entry(
            parent,
            show=show,
            font=("Arial", 15),
            relief="solid",
            bd=1,
        )
        return entry
    def _link_button(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=parent["bg"],
            fg=self.colors["muted"],
            activebackground=parent["bg"],
            activeforeground=self.colors["muted"],
            relief="flat",
            takefocus=0,
            font=("Arial", 12, "bold"),
            cursor="hand2",
        )

    def _validate_required_fields(self, username_entry, password_entry, error_label):
        username = username_entry.get().strip()
        password = password_entry.get()
        if not username and not password:
            error_label.config(text="Enter a username and password.")
            username_entry.focus_set()
            return False
        if not username:
            error_label.config(text="Username is required.")
            username_entry.focus_set()
            return False
        if not password:
            error_label.config(text="Password is required.")
            password_entry.focus_set()
            return False
        error_label.config(text="")
        return True

    def show_home(self):
        self.current_user = None

        def build(frame):
            panel = self._center_panel(frame, max_width=980)
            content = tk.Frame(panel, bg=self.colors["panel"])
            content.grid(row=0, column=0, sticky="nsew", padx=56, pady=52)
            content.columnconfigure(0, weight=1)


            self._label(content, "Emotion Recognition System", 32, "bold").grid(row=1, column=0, pady=(0, 18))
            self._label(
                content,
                "Try the system quickly without saving data, or sign in to use "
                "saved facial landmark coordinates.",
                15,
                color=self.colors["muted"],
            ).grid(row=2, column=0, pady=(0, 40), sticky="ew")

            actions = tk.Frame(content, bg=self.colors["panel"])
            actions.grid(row=3, column=0, pady=(0, 22))
            self._button(actions, "Try", self.start_try_mode, True, text_color="#000000").grid(row=0, column=0, padx=10, sticky="ew")
            self._button(actions, "Sign In", self.show_login, False, text_color="#000000").grid(row=0, column=1, padx=10, sticky="ew")
            self._button(content, "Exit", self.shutdown, False, danger=True, outline=True).grid(row=4, column=0, pady=(6, 0))

        self._set_screen(build)

    def show_login(self):
        def build(frame):
            panel = self._center_panel(frame, max_width=620)
            content = tk.Frame(panel, bg=self.colors["panel"])
            content.grid(row=0, column=0, sticky="nsew", padx=54, pady=46)
            content.columnconfigure(0, weight=1)

            self._label(content, "Sign In", 28, "bold").grid(row=0, column=0, pady=(10, 28), sticky="ew")

            tk.Label(
                content,
                text="Username",
                bg=self.colors["panel"],
                fg=self.colors["text"],
                font=("Arial", 13, "bold")
            ).grid(row=1, column=0, sticky="w")

            username_entry = self._entry(content)
            username_entry.grid(row=2, column=0, sticky="ew", pady=(8, 18), ipady=8)

            tk.Label(
                content,
                text="Password",
                bg=self.colors["panel"],
                fg=self.colors["text"],
                font=("Arial", 13, "bold")
            ).grid(row=3, column=0, sticky="w")

            password_entry = self._entry(content, show="*")
            password_entry.grid(row=4, column=0, sticky="ew", pady=(8, 22), ipady=8)

            error_label = tk.Label(
                content,
                text="",
                bg=self.colors["panel"],
                fg=self.colors["danger"],
                font=("Arial", 12, "bold")
            )
            error_label.grid(row=5, column=0, pady=(0, 14), sticky="ew")

            def submit():
                if not self._validate_required_fields(username_entry, password_entry, error_label):
                    return
                try:
                    self.current_user = authenticate_user(username_entry.get(), password_entry.get())
                    self.show_dashboard()
                except InvalidCredentialsError as exc:
                    error_label.config(text=str(exc))

            username_entry.focus_set()
            username_entry.bind("<Return>", lambda event: password_entry.focus_set())
            password_entry.bind("<Return>", lambda event: submit())

            self._button(
                content,
                "Sign In",
                submit,
                True,
                outline=True,
                text_color="#000000"
            ).grid(row=6, column=0, sticky="ew", pady=(0, 12))

            self._button(
                content,
                "Create Account",
                self.show_register,
                False,

                text_color="#000000"
            ).grid(row=7, column=0, sticky="ew")

            self._button(content, "Back", self.show_home, text_color="#000000").grid(row=8, column=0, pady=(18, 0))

        self._set_screen(build)

    def show_register(self):
        def build(frame):
            panel = self._center_panel(frame, max_width=620)
            content = tk.Frame(panel, bg=self.colors["panel"])
            content.grid(row=0, column=0, sticky="nsew", padx=54, pady=46)
            content.columnconfigure(0, weight=1)

            self._label(content, "Create Account", 28, "bold").grid(row=0, column=0, pady=(10, 28), sticky="ew")

            tk.Label(content, text="Username", bg=self.colors["panel"], fg=self.colors["text"],
                     font=("Arial", 13, "bold")).grid(row=1, column=0, sticky="w")

            username_entry = self._entry(content)
            username_entry.grid(row=2, column=0, sticky="ew", pady=(8, 18), ipady=8)

            tk.Label(content, text="Password", bg=self.colors["panel"], fg=self.colors["text"],
                     font=("Arial", 13, "bold")).grid(row=3, column=0, sticky="w")

            password_entry = self._entry(content, show="*")
            password_entry.grid(row=4, column=0, sticky="ew", pady=(8, 22), ipady=8)

            error_label = tk.Label(content, text="", bg=self.colors["panel"], fg=self.colors["danger"],
                                   font=("Arial", 12, "bold"))
            error_label.grid(row=5, column=0, pady=(0, 14), sticky="ew")

            def submit():
                if not self._validate_required_fields(username_entry, password_entry, error_label):
                    return
                try:
                    self.current_user = create_user(username_entry.get(), password_entry.get())
                    self.show_dashboard()
                except (UserAlreadyExistsError, ValueError) as exc:
                    error_label.config(text=str(exc))

            username_entry.focus_set()
            username_entry.bind("<Return>", lambda event: password_entry.focus_set())
            password_entry.bind("<Return>", lambda event: submit())

            self._button(content, "Create Account", submit, True, text_color="#000000", outline=True,).grid(
                row=6, column=0, sticky="ew", pady=(0, 12)
            )

            self._button(content, "Already Have an Account", self.show_login, False).grid(
                row=7, column=0, sticky="ew"
            )

            self._button(content, "Back", self.show_home, text_color="#000000").grid(row=8, column=0, pady=(18, 0))

        self._set_screen(build)

    def show_dashboard(self):
        if self.current_user is None:
            self.show_home()
            return

        def build(frame):
            panel = self._center_panel(frame, max_width=800)
            content = tk.Frame(panel, bg=self.colors["panel"])
            content.grid(row=0, column=0, sticky="nsew", padx=54, pady=52)
            content.columnconfigure(0, weight=1)

            self._label(content, f"Account: {self.current_user['username']}", 28, "bold").grid(row=0, column=0, pady=(22, 18))

            has_face = bool(self.current_user.get("face_coordinates"))
            status = (
                "Facial landmark coordinates are already saved. Recapture is not required."
                if has_face else
                "Facial landmark coordinates are not saved yet. On the first run, the system will capture your face through the camera."
            )
            status_card = tk.Frame(content, bg=self.colors["surface"], highlightbackground=self.colors["border"], highlightthickness=1)
            status_card.grid(row=1, column=0, sticky="ew", pady=(0, 32))
            self._label(status_card, status, 14, color=self.colors["muted"]).pack(fill="x", padx=24, pady=22)

            self._button(content, "Start Recognition", self.start_authorized_mode, True, text_color="#000000").grid(row=2, column=0, sticky="ew", pady=(0, 14))
            self._button(content, "Recapture Face", self.recapture_face_profile, False, outline=True, text_color="#000000").grid(row=3, column=0, sticky="ew", pady=(0, 12))
            self._button(content, "Sign Out", self.show_home, False, outline=True, text_color="#000000").grid(row=4, column=0, sticky="ew", pady=(0, 12))

        self._set_screen(build)

    def show_launching(self, text):
        def build(frame):
            panel = self._center_panel(frame, max_width=760)
            content = tk.Frame(panel, bg=self.colors["panel"])
            content.grid(row=0, column=0, sticky="nsew", padx=54, pady=52)
            content.columnconfigure(0, weight=1)

            self._label(content, text, 28, "bold").grid(row=0, column=0, pady=(110, 18))
            self._label(
                content,
                "A camera window will open. To finish, press q or Esc in the recognition window.",
                14,
                color=self.colors["muted"],
            ).grid(row=1, column=0, sticky="ew")

        self._set_screen(build)

    def start_try_mode(self):
        self.show_launching("Starting try mode...")
        self.after(
            100,
            lambda: self._run_busy(
                lambda: run_emotion_recognition(save_outputs=False),
                self.show_home,
            ),
        )

    def _run_busy(self, action, on_done):
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            action()
        except CameraUnavailableError as exc:
            messagebox.showerror("Error", str(exc))
        except FaceNotFoundError as exc:
            messagebox.showerror("Error", str(exc))
        except Exception as exc:
            messagebox.showerror("Error", f"Could not start recognition: {exc}")
        finally:
            self.config(cursor="")
            on_done()

    def start_authorized_mode(self):
        if self.current_user is None:
            self.show_home()
            return

        def action():
            user = get_user(self.current_user["id"])
            if user is None:
                messagebox.showerror("Error", "User not found.")
                return
            face_profile = parse_face_profile(user.get("face_coordinates"))

            if face_profile is None:
                face_profile = capture_face_profile()
                update_face_coordinates(
                    user["id"],
                    serialize_face_profile(face_profile),
                )
                user = get_user(user["id"])

            self.current_user = user
            run_emotion_recognition(
                face_profile=face_profile,
                save_outputs=True,
                session_owner=self.current_user["username"],
            )

        self.show_launching("Starting recognition...")
        self.after(100, lambda: self._run_busy(action, self.show_dashboard))

    def recapture_face_profile(self):
        if self.current_user is None:
            self.show_home()
            return

        def action():
            user = get_user(self.current_user["id"])
            if user is None:
                messagebox.showerror("Error", "User not found.")
                return

            face_profile = capture_face_profile()
            update_face_coordinates(
                user["id"],
                serialize_face_profile(face_profile),
            )
            self.current_user = get_user(user["id"])
            messagebox.showinfo("Done", "Facial landmark coordinates have been updated.")

        self.show_launching("Recapturing face...")
        self.after(100, lambda: self._run_busy(action, self.show_dashboard))


def main():
    app = EmotionRecognitionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
