% Open Quantum Institute document
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{institution}[2025/01/01 Custom Institution Class with Header/Footer Graphics]
% Load base class
\LoadClass[12pt]{article}

% Required packages
\RequirePackage{fancyhdr}
\RequirePackage{graphicx}
\RequirePackage{geometry}
\RequirePackage{xcolor}

\newcommand{\subtitle}[1]{\def\@subtitle{#1}}
\providecommand{\@subtitle}{} 

% Page settings
\geometry{top=1.5in,        % Increase top margin to make space
  headheight=20pt,  % Set enough space for image height
  headsep=20pt,     % Space between header and text
  margin=1in,        % Overall margin
  }
\fancypagestyle{plain}
\fancyhf{} % Clear header and footer
\renewcommand{\headrulewidth}{0pt}
% Header graphics
\fancyhead[L]{%
  \begin{picture}(0,0)
    \put(-50,-20){\includegraphics[height = 0.72in]{images/oqi_logo.png}}
  \end{picture}
}
\fancyhead[R]{%
  \begin{picture}(0,0)
    \put(-230,-10){\includegraphics[height = 0.34in]{images/oqi_text.png}}
  \end{picture}
}
% Footer graphic
\fancyfoot[R]{%
 \begin{picture}(0,0)
    \put(-536.5,-35){\makebox[\paperwidth]{%
      \includegraphics[width=\paperwidth]{images/footer.jpg}%
    }}
  \end{picture}
}

\renewcommand{\maketitle}{%
  \thispagestyle{fancy}% apply fancy to title page
  \vspace*{0.8in}
  \begin{center}
  {\LARGE\bfseries \@title \par}
  \vspace{0.5em}
  {\large\bfseries \@subtitle \par}  % italic subtitle here
    \vspace{0.5em}
  {\large \@author \par}
  \vspace{0.5em}
  {\@date \par}
  \vspace{1em}
  \end{center}
  \vspace{1cm}
}

% Redefine \maketitle
%\renewcommand{\maketitle}{\institutiontitle}

% Allow user-defined metadata
\newcommand{\institution}[1]{\def\@institution{#1}}
\pagestyle{fancy}
\endinput