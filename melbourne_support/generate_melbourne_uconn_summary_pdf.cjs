const fs = require("fs");
const path = require("path");
const { PDFDocument, StandardFonts, rgb } = require(
  "/Users/ding/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pdf-lib",
);

const inputPath = path.resolve(
  __dirname,
  "melbourne_uconn_course_summary.md",
);
const outputPath = path.resolve(
  __dirname,
  "Melbourne_UConn_course_summary.pdf",
);

const pageWidth = 612;
const pageHeight = 792;
const marginX = 54;
const marginTop = 56;
const marginBottom = 56;
const contentWidth = pageWidth - marginX * 2;

function wrapText(text, font, fontSize, maxWidth, firstLinePrefix = "", restPrefix = "") {
  const words = text.trim().split(/\s+/).filter(Boolean);
  const lines = [];
  let current = firstLinePrefix;

  for (const word of words) {
    const candidate = current.trim().length === firstLinePrefix.trim().length
      ? `${current}${word}`
      : `${current} ${word}`;

    if (font.widthOfTextAtSize(candidate, fontSize) <= maxWidth) {
      current = candidate;
      continue;
    }

    if (current.trim().length > 0) {
      lines.push(current);
      current = `${restPrefix}${word}`;
      continue;
    }

    lines.push(candidate);
    current = restPrefix;
  }

  if (current.trim().length > 0) {
    lines.push(current);
  }

  return lines.length > 0 ? lines : [firstLinePrefix.trimEnd()];
}

function isSectionHeading(line) {
  return (
    line.length > 0 &&
    !line.startsWith("# ") &&
    !line.startsWith("- ") &&
    !line.startsWith("  ") &&
    !line.includes("://") &&
    !line.includes(": ") &&
    line === line.trim()
  );
}

async function main() {
  const source = fs.readFileSync(inputPath, "utf8");
  const pdfDoc = await PDFDocument.create();
  const fontRegular = await pdfDoc.embedFont(StandardFonts.Helvetica);
  const fontBold = await pdfDoc.embedFont(StandardFonts.HelveticaBold);

  let page = pdfDoc.addPage([pageWidth, pageHeight]);
  let y = pageHeight - marginTop;

  const addPage = () => {
    page = pdfDoc.addPage([pageWidth, pageHeight]);
    y = pageHeight - marginTop;
  };

  const ensureSpace = (heightNeeded) => {
    if (y - heightNeeded < marginBottom) {
      addPage();
    }
  };

  const drawWrappedBlock = ({
    text,
    font,
    fontSize,
    lineHeight,
    color = rgb(0.12, 0.12, 0.12),
    firstLinePrefix = "",
    restPrefix = "",
    spacingAfter = 6,
  }) => {
    const lines = wrapText(
      text,
      font,
      fontSize,
      contentWidth,
      firstLinePrefix,
      restPrefix,
    );

    ensureSpace(lines.length * lineHeight + spacingAfter);

    for (const line of lines) {
      page.drawText(line, {
        x: marginX,
        y,
        size: fontSize,
        font,
        color,
      });
      y -= lineHeight;
    }

    y -= spacingAfter;
  };

  const lines = source.split(/\r?\n/);

  for (const rawLine of lines) {
    const line = rawLine.replace(/\t/g, "    ");

    if (line.trim() === "") {
      y -= 8;
      continue;
    }

    if (line.startsWith("# ")) {
      drawWrappedBlock({
        text: line.slice(2),
        font: fontBold,
        fontSize: 20,
        lineHeight: 24,
        color: rgb(0.05, 0.22, 0.45),
        spacingAfter: 10,
      });
      continue;
    }

    if (isSectionHeading(line)) {
      drawWrappedBlock({
        text: line,
        font: fontBold,
        fontSize: 13,
        lineHeight: 16,
        color: rgb(0.15, 0.15, 0.15),
        spacingAfter: 4,
      });
      continue;
    }

    if (line.startsWith("- ")) {
      drawWrappedBlock({
        text: line.slice(2),
        font: fontRegular,
        fontSize: 11,
        lineHeight: 14,
        firstLinePrefix: "• ",
        restPrefix: "  ",
        spacingAfter: 2,
      });
      continue;
    }

    if (line.startsWith("  ")) {
      drawWrappedBlock({
        text: line.trim(),
        font: fontRegular,
        fontSize: 11,
        lineHeight: 14,
        firstLinePrefix: "  ",
        restPrefix: "  ",
        spacingAfter: 2,
      });
      continue;
    }

    drawWrappedBlock({
      text: line,
      font: fontRegular,
      fontSize: 11,
      lineHeight: 14,
      spacingAfter: 4,
    });
  }

  const pages = pdfDoc.getPages();
  pages.forEach((pdfPage, index) => {
    pdfPage.drawText(`Page ${index + 1} of ${pages.length}`, {
      x: pageWidth - marginX - 70,
      y: 24,
      size: 9,
      font: fontRegular,
      color: rgb(0.45, 0.45, 0.45),
    });
  });

  const pdfBytes = await pdfDoc.save();
  fs.writeFileSync(outputPath, pdfBytes);
  console.log(outputPath);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
